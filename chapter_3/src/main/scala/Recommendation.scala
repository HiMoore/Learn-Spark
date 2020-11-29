import scala.collection
import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SparkSession, DataFrame, Dataset}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import org.apache.log4j.{Level, Logger}



object Recommendation {

    Logger.getLogger("org").setLevel(Level.WARN)
    val logg = Logger.getLogger(this.getClass)

    def main(args: Array[String]): Unit = {

        logg.warn("This is warn message.")
        logg.error("This is warn message.")

        val conf = new SparkConf().setMaster("local[2]").setAppName("Chapter 3: Recommendation")
        // val spark = SparkSession.builder().appName("Chapter 3: Recommendation").getOrCreate()
        val spark = SparkSession.builder().config(conf=conf).enableHiveSupport().getOrCreate()

        spark.sparkContext.setCheckpointDir("hdfs://192.168.2.104:9000/tmp")
        import spark.implicits._
        println("\n==================== Spark Job: Read DataFile ! ====================\n")
        val base_dir = "hdfs://192.168.2.104:9000/user/bdhysfas/data/profiledata_06-May-2005/"
        val rawUserArtistData = spark.read.textFile(base_dir + "user_artist_data.txt")
        val rawArtistData = spark.read.textFile(base_dir + "artist_data.txt")
        val rawArtistAlias = spark.read.textFile(base_dir + "artist_alias.txt")

        // pre-process stage 1
        println("\n==================== Spark Job: Data Pre-Process ! ====================\n")
        val userArtistDF = rawUserArtistData.map{ line =>
            val Array(user, artist, _*) = line.split(' ')
            (user.toInt, artist.toInt)
        }.toDF("user", "artist")
        userArtistDF.agg( min("user"), max("user"), min("artist"), max("artist") ).show()
        
        val artistByID = rawArtistData.flatMap{ line => 
            val (id, name) = line.span(_ != '\t')
            if (name.isEmpty) None
            else {
                try {
                    Some( (id.toInt, name.trim) )
                } catch {
                    case _: NumberFormatException => None
                }
            }
        }.toDF("id", "name")
        println( artistByID.count() )

        val artistAlias_T = rawArtistAlias.flatMap { line =>
            val Array(artist, alias) = line.split('\t')
            if (artist.isEmpty) None
            else {
                Some( (artist.toInt, alias.toInt) )
            }
        }.collect().toMap

        val artistAlias = collection.Map(artistAlias_T.toSeq: _*) 
        

        // pre-process stage 2
        val runCommendation = new RunCommendation(spark)
        val bArtistAlias = spark.sparkContext.broadcast(artistAlias)
        val trainData = runCommendation.buildCounts( rawUserArtistData, bArtistAlias )
        trainData.persist(StorageLevel.MEMORY_AND_DISK_SER)
        trainData.show(5)

        println("\n==================== Spark Job: Training Model ! ====================\n")
        runCommendation.model( trainData, bArtistAlias, artistByID )
        // println("\n==================== Spark Job: Evaluate Model ! ====================\n")
        // runCommendation.evaluate( trainData, bArtistAlias, artistByID )
        println("\n==================== Spark Job: Recommend for one user with trained Model ! ====================\n")
        val alsModel = runCommendation.recommend( trainData, bArtistAlias, artistByID )
        println("\n==================== Spark Job: Recommend for some users with trained Model ! ====================\n")
        runCommendation.toRecommend(trainData, alsModel )

        trainData.unpersist()
        spark.stop()
        println("\n==================== Spark Job Run Successfully! ====================\n")
    }
}




class RunCommendation(private val spark: SparkSession) {

    import spark.implicits._

    def buildCounts( rawUserArtistData: Dataset[String], bArtistAlias: Broadcast[Map[Int, Int]] ): DataFrame = {
        rawUserArtistData.map { line => 
            val Array( userId, artistID, count ) = line.split(" ").map(_.toInt)
            val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
            ( userId, finalArtistID, count )
        }.toDF("user", "artist", "count")
    }


    def model( trainData: DataFrame, bArtistAlias: Broadcast[Map[Int, Int]], artistByID: DataFrame ): Unit = {
        val model = new ALS().setSeed(Random.nextLong()).setImplicitPrefs(true).
                setRank(10).setRegParam(0.01).setAlpha(1.0).setMaxIter(5).
                setUserCol("user").setItemCol("artist").setRatingCol("count").setPredictionCol("prediction").
                fit(trainData)
        model.userFactors.select("features").show(truncate=false)

        println(s"\n-------------------- Spark Job: Recommendate for one user with trained model --------------------\n")
        val userID = 2093760
        val existingArtistIDs = trainData.filter($"user" === userID).select("artist").as[Int].collect()
        artistByID.filter( $"id" isin (existingArtistIDs: _*) ).show(5, false)
        val topRecommendations = makeRecommendations(model, userID, 5)
        topRecommendations.show(5, false)
        val recommendedArtistIDs = topRecommendations.select("artist").as[Int].collect()
        artistByID.filter($"id" isin (recommendedArtistIDs: _*)).show(5, false)

        model.userFactors.unpersist()
        model.itemFactors.unpersist()
    }


    def makeRecommendations(model: ALSModel, userID: Int, howMany: Int): DataFrame = {
        val toRecommend = model.itemFactors.select($"id".as("artist")).withColumn("user", lit(userID))
        model.transform(toRecommend).select("artist", "prediction").orderBy($"prediction".desc).limit(howMany)
    }


    def evaluate( allData: DataFrame, bArtistAlias: Broadcast[Map[Int, Int]], artistByID: DataFrame ): Unit = {
        val Array(trainData, cvData) = allData.randomSplit( Array(0.9, 0.1) )
        trainData.persist(StorageLevel.MEMORY_AND_DISK_SER)
        cvData.persist(StorageLevel.MEMORY_AND_DISK_SER)

        val allArtistIDs = allData.select("artist").as[Int].distinct().collect()
        val bAllArtistIDs = spark.sparkContext.broadcast( allArtistIDs )
        // evaluate the most listened algo
        val mostListenedAUC = areaUnderCurve( cvData, bAllArtistIDs, predictMostListened(trainData) )
        println(s"\n-------------------- Most Listened AUC: $mostListenedAUC --------------------\n")

        val evaluations = 
            for ( rank <- Seq(5, 30); regParam <- Seq(1.0, 0.0001); alpha <- Seq(1.0, 40.0) ) 
            yield {
                val model = new ALS().setSeed(Random.nextLong()).setImplicitPrefs(true).
                        setRank(rank).setRegParam(regParam).setAlpha(alpha).setMaxIter(20).
                        setUserCol("user").setItemCol("artist").setRatingCol("count").setPredictionCol("prediction").
                        fit(trainData)
                val auc = areaUnderCurve( cvData, bAllArtistIDs, model.transform )
                model.userFactors.unpersist()
                model.itemFactors.unpersist()

                ( auc, (rank, regParam, alpha) )
                println(s"\n ---------- auc: $auc, rank: $rank, regParam: $regParam, alpha: $alpha -----------  \n")
            }
        println("\n -------------------- start print auc: (rank, regParam, alpha) -------------------- \n")
        evaluations.sorted.reverse.foreach(println)
        println("\n -------------------- end print auc: (rank, regParam, alpha) -------------------- \n")
        trainData.unpersist()
        cvData.unpersist()

    }


    def predictMostListened(trainData: DataFrame)(allData: DataFrame): DataFrame = {
        val listenCounts = trainData.groupBy("artist").agg( sum("count").as("prediction") ).select("artist", "prediction")
        allData.join(listenCounts, Seq("artist"), "left_outer").select("user", "artist", "prediction")
    }


    def areaUnderCurve( positiveData: DataFrame, bAllArtistIDs: Broadcast[Array[Int]], predictFunction: (DataFrame => DataFrame) ): Double = {
        val positivePredictions = predictFunction( positiveData.select("user", "artist")).withColumnRenamed("prediction", "positivePrediction")
        val negativeData = positiveData.select("user", "artist").as[(Int, Int)].
            groupByKey{ case (user, _) => user }.flatMapGroups { case (userID, userIDAndPosArtistIDs) => 
                val random = new Random()
                val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
                val negative = new ArrayBuffer[Int]()
                val allArtistIDs = bAllArtistIDs.value
                var i = 0
                while ( i<allArtistIDs.length && negative.size<posItemIDSet.size ) {
                    val artistID = allArtistIDs( random.nextInt(allArtistIDs.length) )
                    if (!posItemIDSet.contains(artistID)) {
                        negative += artistID
                    }
                    i += 1
                }
                negative.map( artistID => (userID, artistID) )
            }.toDF("user", "artist")

        val negativePredictions = predictFunction(negativeData).withColumnRenamed("prediction", "negativePrediction")
        val joinedPredictions = positivePredictions.join( negativePredictions, "user" ).select("user", "positivePrediction", "negativePrediction")
        joinedPredictions.cache()

        val allCounts = joinedPredictions.groupBy("user").agg( count(lit("1")).as("total") ).select("user", "total")
        val correctCounts = joinedPredictions.filter($"positivePrediction" > $"negativePrediction").
            groupBy("user").agg( count("user").as("correct") ).select("user", "correct")
        val meanAUC = allCounts.join( correctCounts, Seq("user"), "left_outer" ).
            select( $"user", ( coalesce($"correct", lit(10)) / $"total" ).as("auc") ).
            agg( mean("auc") ).as[Double].first()
        joinedPredictions.unpersist()

        meanAUC
    }


    def recommend( allData: DataFrame, bArtisAlias: Broadcast[Map[Int, Int]], artistByID: DataFrame ): ALSModel = {
        val model = new ALS().setSeed(Random.nextLong()).setImplicitPrefs(true).
            setRank(10).setRegParam(1.0).setAlpha(40.0).setMaxIter(20).
            setUserCol("user").setItemCol("artist").setRatingCol("count").setPredictionCol("prediction").
            fit(allData)
        val userID = 2093760
        println("\n -------------------- Recommend with user-defined makeRecommendation function -------------------- \n")
        val topRecommendations = makeRecommendations(model, userID, 5)
        val recommendedArtistIDs = topRecommendations.select("artist").as[Int].collect()
        artistByID.join( spark.createDataset(recommendedArtistIDs).toDF("id"), "id" ).select("name").show(5, false)
        artistByID.filter( $"id" isin (recommendedArtistIDs: _*) ).show(5, false)
        println("\n -------------------- Recommend with Spark-Owned makeRecommendation function -------------------- \n")
        val topRecommendation2 = model.recommendForUserSubset( allData.filter($"user" === userID).select("user"), 5 )
        val recommenedArtists = topRecommendation2.select( explode($"recommendations") ).
            withColumn("artist", $"col".getField("artist")).withColumn("rating", $"col".getField("rating")).
            drop("col")
        val recommendedArtistIDs2 = recommenedArtists.select("artist").as[Int].collect()
        artistByID.filter( $"id" isin (recommendedArtistIDs2: _*) ).show(5, false)
        
        model.userFactors.unpersist()
        model.itemFactors.unpersist()
        
        model
    }


    def toRecommend( allData: DataFrame, model: ALSModel ): Unit = {
        val someUsers = allData.select("user").as[Int].distinct().take(100)
        val someRecommendations = someUsers.map( userID => (userID, makeRecommendations(model, userID, 5)) )
        someRecommendations.foreach { case (userID, recsDF) =>
            val recommendedArtists = recsDF.select("artist").as[Int].collect()
            println(s"\n$userID -> ${recommendedArtists.mkString(",")}\n")
        }
    }
}