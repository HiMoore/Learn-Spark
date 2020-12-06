// Scala with SBT
/*
@File    :   RandomDecisionForest.scala
@Time    :   2020/11/30 20:49:24
@Author  :   Chen Shuai 
@Version :   1.0
@Contact :   qiranoo@126.com
*/

import scala.util.Random

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel, RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.tuning.{TrainValidationSplit, TrainValidationSplitModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}

import org.apache.log4j.{Level, Logger}




object RandomDecisionForest {
    
    Logger.getLogger("org").setLevel(Level.WARN)
    val logg = Logger.getLogger(this.getClass)
    
    def main( args: Array[String] ): Unit = {

        val conf = new SparkConf()
            .set("spark.driver.cores","4")  //设置driver的CPU核数
            .set("spark.driver.maxResultSize","2g") //设置driver端结果存放的最大容量，这里设置成为2G，超过2G的数据,job就直接放弃，不运行了
            .set("spark.driver.memory","3g")  //driver给的内存大小
            .set("spark.executor.memory","1g")// 每个executor的内存
            .set("spark.submit.deployMode","client")  //spark 任务提交模式，线上使用cluster模式，开发使用client模式
            .set("spark.worker.timeout" ,"500") //基于standAlone模式下提交任务，worker的连接超时时间
            .set("spark.cores.max" , "4")  //基于standAlone和mesos模式下部署，最大的CPU和数量
            .set("spark.rpc.askTimeout" , "600s")  //spark任务通过rpc拉取数据的超时时间
            .set("spark.locality.wait" , "5s") //每个task获取本地数据的等待时间，默认3s钟，如果没获取到，依次获取本进程，本机，本机架数据
            .set("spark.task.maxFailures" , "5")  //允许最大失败任务数，根据自身容错情况来定
            .set("spark.serializer" ,"org.apache.spark.serializer.KryoSerializer")  //配置序列化方式
            // Spark-Streaming
            .set("spark.streaming.kafka.maxRatePerPartition" , "5000")  //使用directStream方式消费kafka当中的数据，获取每个分区数据最大速率
            .set("spark.streaming.backpressure.enabled" , "true")  //开启sparkStreaming背压机制，接收数据的速度与消费数据的速度实现平衡
            .set("spark.streaming.backpressure.initialRate" , "500") 
            //  .set("spark.streaming.backpressure.pid.minRate","10")
            .set("spark.driver.host", "localhost")  //配置driver地址
            //shuffle相关参数调优开始
            .set("spark.reducer.maxSizeInFlight","128m")  //reduceTask拉取map端输出的最大数据量，调整太大有OOM的风险
            .set("spark.shuffle.compress","true")  //开启shuffle数据压缩
            .set("spark.default.parallelism","4")  //设置任务的并行度
            .set("spark.files.fetchTimeout","120s")  //设置文件获取的超时时间
            //网络相关参数
            .set("spark.rpc.message.maxSize","128")  //RPC拉取数据的最大数据量，单位M
            .set("spark.network.timeout","120s")  //网络超时时间设置
            .set("spark.scheduler.mode","FAIR")  //spark 任务调度模式  使用 fair公平调度
            //spark任务资源动态划分  https://spark.apache.org/docs/2.3.0/job-scheduling.html#configuration-and-setup
            .set("spark.dynamicAllocation.enabled","true")
            .set("spark.shuffle.service.enabled","true")
            .set("spark.dynamicAllocation.executorIdleTimeout","120s")  //executor空闲时间超过这个值，该executor就会被回收
            .set("spark.dynamicAllocation.minExecutors","0")  //最少的executor个数
            .set("spark.dynamicAllocation.maxExecutors","2")  //最大的executor个数  根据自己实际情况调整
            .set("spark.dynamicAllocation.initialExecutors","1")//初始executor个数
            .set("spark.dynamicAllocation.schedulerBacklogTimeout","5s")  //pending 状态的task时间，过了这个时间继续pending ，申请新的executor
            // 推测执行
            .set("spark.speculation", "true")   //开启推测执行
            .set("spark.speculation.interval", "100s") 
            .set("spark.speculation.quantile","0.9")
            .setMaster("local[*]")
            .setAppName("Chapter 4: Decision Tree")

        val spark = SparkSession.builder().config(conf=conf).enableHiveSupport().getOrCreate()
        println("\n==================== Spark Job: Read DataFile ! ====================\n")
        val baseDir = "hdfs://192.168.2.104:9000/user/bdhysfas/data/"
        val dataWithoutHeader = spark.read.option("inferSchema", true).option("header", false).
            csv(baseDir + "covtype.data")
        val colNames = Seq( "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", 
                "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
                ) ++ ( 
                    (0 until 4).map( i => s"Wilderness_Area_$i" )
                ) ++ (
                    (0 until 40).map( i => s"Soil_Type_$i" )
                ) ++ Seq("Cover_Type")
        val data = dataWithoutHeader.toDF( colNames: _* ).withColumn( "Cover_Type", col("Cover_Type").cast("double") )
        data.show(5, false)
        val Array( trainData, testData ) = data.randomSplit( Array(0.9, 0.1) )

        trainData.persist(StorageLevel.MEMORY_AND_DISK_SER)
        testData.persist(StorageLevel.MEMORY_AND_DISK_SER)
        val inputCols = trainData.columns.filter(_ != "Cover_Type")
        val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featureVector")
        val assembledTrainData = assembler.transform(trainData)
        
        println("\n==================== Spark Job: Training Model ! ====================\n")
        val rdf = new RunRDF(spark)
        val decisionTree = rdf.simpleDecisionTree( trainData, assembledTrainData )
        val featureImportances = decisionTree.featureImportances.toArray.zip(inputCols).sorted.reverse
        (0 until 5).map( x => println(featureImportances(x)) )
        val predictions = decisionTree.transform(assembledTrainData)
        predictions.select("Cover_Type", "prediction", "probability").show(5, false)

        println("\n==================== Spark Job: Evaluating Model ! ====================\n")
        rdf.evaluate_model(predictions)
        rdf.classProbabilities(trainData, testData)
        
        // println("\n==================== Spark Job: Optimaze Model ! ====================\n")
        // val validateModel = rdf.optimazeDecisionTree(trainData, testData)

        println("\n==================== Spark Job: Training Model without One-Hot enc ! ====================\n")
        val unencTrainData = rdf.unencodeOneHot(trainData)
        val unencTestData = rdf.unencodeOneHot(testData)
        var testAccuracy = rdf.evaluateCategorical(unencTrainData, unencTestData)
        
        println("\n==================== Spark Job: RandomForest Model ! ====================\n")
        testAccuracy = rdf.evaluateForest(unencTrainData, unencTestData)

        println("\n==================== Spark Job: Stop ! ====================\n")
        trainData.unpersist()
        testData.unpersist()
        spark.stop()


    }
}





class RunRDF(private val spark: SparkSession) {
    
    import spark.implicits._

    def evaluateForest(trainData: DataFrame, testData: DataFrame): Double = {
        val inputCols = trainData.columns.filter( _ != "Cover_Type" )
        val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featureVector")
        val indexer = new VectorIndexer().setMaxCategories(40).setInputCol("featureVector")setOutputCol("indexedVector")
        val classifier = new RandomForestClassifier().setSeed(Random.nextLong()).setLabelCol("Cover_Type").
            setFeaturesCol("indexedVector").setPredictionCol("prediction").setImpurity("entropy").
            setMaxDepth(20).setMaxBins(300)

        val pipeline = new Pipeline().setStages( Array(assembler, indexer, classifier) )
        val paramGrid = new ParamGridBuilder().addGrid( classifier.minInfoGain, Seq(0.0, 0.05) ).
            addGrid( classifier.numTrees, Seq(1, 10) ).build()
        val multiclassEval = new MulticlassClassificationEvaluator().setLabelCol("Cover_Type").
            setPredictionCol("prediction").setMetricName("accuracy")

        val validator = new TrainValidationSplit().setSeed(Random.nextLong()).setEstimator(pipeline).
            setEvaluator(multiclassEval).setEstimatorParamMaps(paramGrid).setTrainRatio(0.9)
        val validatorModel = validator.fit(trainData)
        val bestModel = validatorModel.bestModel
        val forestModel = bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[RandomForestClassificationModel]
        println(forestModel.extractParamMap)
        println(forestModel.getNumTrees)
        forestModel.featureImportances.toArray.zip(inputCols).sorted.reverse.foreach(println)

        val testAccuracy = multiclassEval.evaluate( bestModel.transform(testData) )
        println( testAccuracy )

        testAccuracy
    }


    def evaluateCategorical(trainData: DataFrame, testData: DataFrame): Double = {
        val inputCols = trainData.columns.filter(_ != "Cover_Type")
        val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featureVector")
        val indexer = new VectorIndexer().setMaxCategories(40).setInputCol("featureVector").setOutputCol("indexedVector")
        val classifier = new DecisionTreeClassifier().setSeed(Random.nextLong()).
            setLabelCol("Cover_Type").setFeaturesCol("indexedVector").setPredictionCol("prediction")

        val pipeline = new Pipeline().setStages( Array(assembler, indexer, classifier) )
        val paramGrid = new ParamGridBuilder().addGrid( classifier.impurity, Seq("gini", "entropy") ).
            addGrid( classifier.maxDepth, Seq(1, 20) ).addGrid( classifier.maxBins, Seq(40, 300) ).
            addGrid( classifier.minInfoGain, Seq(0.0, 0.05) ).build()
        val multiclassEval = new MulticlassClassificationEvaluator().setLabelCol("Cover_Type").
            setPredictionCol("prediction").setMetricName("accuracy")
        val validator = new TrainValidationSplit().setSeed(Random.nextLong()).
            setEstimator(pipeline).setEvaluator(multiclassEval).setEstimatorParamMaps(paramGrid).
            setTrainRatio(0.9)
        val validatorModel = validator.fit( trainData )
        
        val bestModel = validatorModel.bestModel
        println( bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap )
        val testAccuracy = multiclassEval.evaluate( bestModel.transform(testData) )
        println( testAccuracy )

        testAccuracy
    }


    def unencodeOneHot(data: DataFrame): DataFrame = {
        val wildernessCols = (0 until 4).map( i => s"Wilderness_Area_$i").toArray
        val widlernessAssembler = new VectorAssembler().setInputCols(wildernessCols).setOutputCol("wilderness")
        val unhotUDF = udf( (vec: Vector) => vec.toArray.indexOf(1.0).toDouble )
        val withWilderness = widlernessAssembler.transform(data).drop(wildernessCols: _*).
            withColumn( "wilderness", unhotUDF($"wilderness") )
        
        val soilCols = (0 until 40).map( i => s"Soil_Type_$i" ).toArray
        val soilAssembler = new VectorAssembler().setInputCols(soilCols).setOutputCol("soil")
        
        soilAssembler.transform(withWilderness).drop(soilCols: _*).
            withColumn("soil", unhotUDF($"soil"))
    }


    def optimazeDecisionTree(trainData: DataFrame, testData: DataFrame): TrainValidationSplitModel = {
        val inputCols = trainData.columns.filter(_ != "Cover_Type")
        val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featureVector")
        val classifier = new DecisionTreeClassifier().setSeed(Random.nextLong()).setLabelCol("Cover_Type").
            setFeaturesCol("featureVector").setPredictionCol("prediction")

        val pipeline = new Pipeline().setStages(Array(assembler, classifier))
        val paramGrid = new ParamGridBuilder().addGrid( classifier.impurity, Seq("gini", "entropy") ).
            addGrid( classifier.maxDepth, Seq(1, 20) ).addGrid( classifier.maxBins, Seq(40, 300) ).
            addGrid( classifier.minInfoGain, Seq(0.0, 0.05) ).build()
        val multiclassEval = new MulticlassClassificationEvaluator().setLabelCol("Cover_Type").
            setPredictionCol("prediction").setMetricName("accuracy")
        val validator = new TrainValidationSplit().setSeed(Random.nextLong()).setEstimator(pipeline).
            setEvaluator(multiclassEval).setEstimatorParamMaps(paramGrid).setTrainRatio(0.9)
        println("-------------------- Training --------------------")
        val validatorModel = validator.fit(trainData)
        println("-------------------- Train Completed --------------------")        
        val paramsAndMetrics = validatorModel.validationMetrics.zip( validatorModel.getEstimatorParamMaps ).sortBy(-_._1)
        paramsAndMetrics.foreach { case (metric, params) => 
            println(metric)
            println(s"$params\n")
        }

        val bestModel = validatorModel.bestModel
        println( bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap )
        println( validatorModel.validationMetrics.max )
        val trainAccuracy = multiclassEval.evaluate(bestModel.transform(trainData))
        val testAccuracy = multiclassEval.evaluate(bestModel.transform(testData))
        println(s"TrainAccuracy: $trainAccuracy, TestAccuracy: $testAccuracy")

        validatorModel
    }


    def classProbabilities(trainData: DataFrame, testData: DataFrame): Double = {
        val trainTotal = trainData.count()
        val trainPrior = trainData.groupBy("Cover_Type").count().orderBy("Cover_Type").
            select("count").as[Double].map(_ / trainTotal).collect()
        val testTotal = testData.count()
        val testPrior = testData.groupBy("Cover_Type").count().orderBy("Cover_Type").
            select("count").as[Double].map(_ / testTotal).collect()
        val accuracy = trainPrior.zip(testPrior).map {
            case (trainProb, cvProb) => trainProb * cvProb
        }.sum

        accuracy
    }


    def evaluate_model(predictions: DataFrame): Unit = {
        val evaluator = new MulticlassClassificationEvaluator().setLabelCol("Cover_Type").setPredictionCol("prediction")
        println( "Accuracy score: %.4f".format( evaluator.setMetricName("accuracy").evaluate(predictions)) )
        println( "f1 score: %.4f".format( evaluator.setMetricName("f1").evaluate(predictions)) )
        val predictionRDD = predictions.select("prediction", "Cover_Type").as[(Double, Double)].rdd
        val multiclassMetrics = new MulticlassMetrics(predictionRDD)
        println( multiclassMetrics.confusionMatrix )

        val confusionMatrix = predictions.groupBy("Cover_Type").pivot("prediction", (1 to 7)).count().na.fill(0.0).orderBy("Cover_Type")
        confusionMatrix.show()
    }


    def simpleDecisionTree(trainData: DataFrame, assembledTrainData: DataFrame): DecisionTreeClassificationModel = {
        val classifier = new DecisionTreeClassifier().setSeed(Random.nextLong()).setLabelCol("Cover_Type").setFeaturesCol("featureVector").setPredictionCol("prediction")
        val model = classifier.fit(assembledTrainData)

        model
    }
}