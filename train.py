from __future__ import print_function

from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import date_format, sin, cos, radians, atan2, month
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

def distance(long1, lat1, long2, lat2):
        radius = 6371
        diff_lat = radians(lat2 - lat1)
        diff_long = radians(long2 - long1)
        a = sin(diff_lat/2)**2 + cos(lat1)*cos(lat2)*sin(diff_long/2)**2
        c = 2*atan2(a**0.5, (1-a)**0.5)
        return radius*c

if __name__ == "__main__":
        SparkContext.setSystemProperty("saprk.executor.memory", "12g")
        spark = SparkSession.builder.appName("RegressionTree").getOrCreate()

        # Load up data as dataframe
        data = spark.read.csv("/Users/rafaelchen/Documents/MapReduce/hw2 decision tree/src/train.csv", header=True)
        # Data preprocessing
        data = data.withColumn("pickup_longitude", data["pickup_longitude"].cast("float")).withColumn("pickup_latitude", data["pickup_latitude"].cast("float")).withColumn("dropoff_longitude", data["dropoff_longitude"].cast("float")).withColumn("dropoff_latitude", data["dropoff_latitude"].cast("float")).withColumn("passenger_count", data["passenger_count"].cast("int")).withColumn("trip_duration", data["trip_duration"].cast("int")).withColumn("pickup_datetime", data["pickup_datetime"].cast("timestamp")).withColumn("dropoff_datetime", data["dropoff_datetime"].cast("timestamp")).withColumn("vendor_id", data["vendor_id"].cast("int"))

        data = data.withColumn("pickup_weekday", date_format("pickup_datetime", "E")).withColumn("pickup_hour", date_format("pickup_datetime", "H")).withColumn("pickup_month", date_format("pickup_datetime", "M"))
        data = data.withColumn("pickup_hour", data["pickup_hour"].cast("int")).withColumn("pickup_month", data["pickup_month"].cast("int"))
        data = data.withColumn("dropoff_weekday", date_format("dropoff_datetime", "E")).withColumn("dropoff_hour", date_format("dropoff_datetime", "H")).withColumn("dropoff_month", date_format("dropoff_datetime", "M"))
        data = data.withColumn("dropoff_hour", data["dropoff_hour"].cast("int")).withColumn("dropoff_month", data["dropoff_month"].cast("int"))
        data = data.withColumn("trip_distance", distance(data.pickup_longitude, data.pickup_latitude, data.dropoff_longitude, data.dropoff_latitude))
        # Data cleaning
        data = data.filter(data["trip_duration"] > 10).filter(data["trip_duration"] < 22*60*60).filter(data["pickup_longitude"] <= -73.75).filter(data["pickup_longitude"] >= -74.03).filter(data["dropoff_longitude"] <= -73.75).filter(data["dropoff_longitude"] >= -74.03).filter(data["pickup_latitude"] <= 40.85).filter(data["pickup_latitude"] >= 40.63).filter(data["dropoff_latitude"] <= 40.85).filter(data["dropoff_latitude"] >= 40.63)
        #data.printSchema()
        assembler = VectorAssembler().setInputCols(["vendor_id", "pickup_longitude", "pickup_latitude", "pickup_hour", "pickup_month", "dropoff_longitude", "dropoff_latitude", "trip_distance", "passenger_count"]).setOutputCol("features")
        df = assembler.setHandleInvalid("skip").transform(data).select("trip_duration", "features")

        featureIndexer = VectorIndexer(inputCol = "features", outputCol = "indexedFeatures", maxCategories = 30).fit(df)
        d = featureIndexer.transform(df)
        trainTest = d.randomSplit([0.8, 0.2])
        traindf = trainTest[0]
        testdf = trainTest[1]

        # Model
        dtr = DecisionTreeRegressor(featuresCol="indexedFeatures", labelCol="trip_duration", impurity="variance")

        # choices of tuning parameters
        dtrparamGrid = (ParamGridBuilder().addGrid(dtr.maxDepth, [10]).build())

        pipeline = Pipeline(stages = [featureIndexer, dtr])

        crossval = CrossValidator(estimator = pipeline, estimatorParamMaps = dtrparamGrid, evaluator = RegressionEvaluator(labelCol = "trip_duration", predictionCol = "prediction", metricName = "rmse"), numFolds = 10)
        model = crossval.fit(traindf)

        predictions = model.transform(testdf).cache()

        predictions.show(25)

        evaluator = RegressionEvaluator(labelCol="trip_duration", predictionCol = "prediction", metricName = "rmse")
        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

        with open("./test.txt", "a") as f:
                f.write("\nHere is the result of RMSE with regression tree: " + str(rmse))

        evaluator2 = RegressionEvaluator(labelCol="trip_duration", predictionCol = "prediction", metricName = "mae")
        mae = evaluator2.evaluate(predictions)
        print("Mean Absolute Error (MAE) on test data = %g" % mae)

        with open("./test.txt", "a") as f:
                f.write("\nHere is the result of MAE with regression tree: " + str(mae))
        """ Here was the part to try enhanced decision tree
        pred = predictions.select("prediction").dropDuplicates()
        for leaf_value in pred.collect():
                input_df = predictions.filter(predictions["prediction"] == leaf_value)
                (train_input_df, test_input_df) = input_df.randomSplit([0.8, 0.2])
                lr = LinearRegression()
                model2 = lr.fit(train_input_df)
                predictions2 = model2.transform(test_input_df)
                predictions2.show(25)
                rmse2 = evaluator.evaluate(predictions2)
                with open("./test.txt", "a") as f:
                        f.write("\n Here is the result of RMSE for current leaf" + str(rmse2))

                mae2 = evaluator2.evaluate(predictions2)
                with open("./test.txt", "a") as f:
                        f.write("\n Here is the result of MAE for current leaf" + str(mae2))
        """
        spark.stop()
