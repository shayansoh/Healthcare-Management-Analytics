// Databricks notebook source
//import all required libraries 
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder}
import org.apache.spark.sql
import org.apache.spark.ml.linalg.{Matrix,Vector,Vectors,SparseVector}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Column
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel,XGBoostClassifier}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

// COMMAND ----------

// data import
val df = spark.read.option("header","true").option("inferSchema","true").format("csv").csv("dbfs:/FileStore/tables/train_data.csv")


// COMMAND ----------

// reducing total number of classes by combining bin sizes
val df_clean = df.withColumn("Stay", regexp_replace($"Stay", "41-50", "41-60"))
               .withColumn("Stay", regexp_replace($"Stay", "51-60", "41-60"))
               .withColumn("Stay", regexp_replace($"Stay", "61-70", "More than 61 Days"))
               .withColumn("Stay", regexp_replace($"Stay", "71-80", "More than 61 Days"))
               .withColumn("Stay", regexp_replace($"Stay", "81-90", "More than 61 Days"))
               .withColumn("Stay", regexp_replace($"Stay", "91-100", "More than 61 Days"))
               .withColumn("Stay", regexp_replace($"Stay", "More than 100 Days", "More than 61 Days"))
               .dropDuplicates()
               .na.drop()

// COMMAND ----------

// downsampling classes
val df_0_10 = df_clean.where("Stay = '0-10'")
val df_11_20 =  df_clean.where("Stay = '11-20'").sample(false,0.3)
val df_21_30 = df_clean.where("Stay = '21-30'").sample(false,0.27)
val df_31_40 = df_clean.where("Stay = '31-40'").sample(false,0.43)
val df_41_60 = df_clean.where("Stay = '41-60'").sample(false,0.505)
val df_61_plus = df_clean.where("Stay = 'More than 61 Days'").sample(false,0.867)
val df_final = df_0_10.union(df_11_20).union(df_21_30).union(df_31_40).union(df_41_60).union(df_61_plus)

val Array(train_tmp, test_tmp) = df_final.randomSplit(Array(0.7, 0.3), 42)



// COMMAND ----------

//feature engineering
// creating a feature that will show which patients went to their own city's hospital
val train_same_city = train_tmp.withColumn("same_city_patient", when(col("City_Code_Patient") === col("City_Code_Hospital"), 1).otherwise(0))
val test_same_city = test_tmp.withColumn("same_city_patient", when(col("City_Code_Patient") === col("City_Code_Hospital"), 1).otherwise(0))

// creating a feature that can show how many times does a particular patient visit
val temp_train_cnt = train_same_city.groupBy("patientid").count().withColumnRenamed("count", "times_visited").withColumnRenamed("patientid", "patientid_")
val temp_test_cnt = test_same_city.groupBy("patientid").count().withColumnRenamed("count", "times_visited").withColumnRenamed("patientid", "patientid_")

val train_times_visited = train_same_city.join(temp_train_cnt,$"patientid" === $"patientid_", "left").drop("patientid_")
val test_times_visited = test_same_city.join(temp_test_cnt,$"patientid" === $"patientid_", "left").drop("patientid_")

// creating a feature that shows admission deposit per patient
val temp_train_sum_deposit = train_times_visited.groupBy("patientid").agg(sum("Admission_Deposit").as("total_admission_deposit")).withColumnRenamed("patientid", "patientid_")
val temp_test_sum_deposit = test_times_visited.groupBy("patientid").agg(sum("Admission_Deposit").as("total_admission_deposit")).withColumnRenamed("patientid", "patientid_")

val train_bill_per_patient = train_times_visited.join(temp_train_sum_deposit,$"patientid" === $"patientid_", "left").drop("patientid_")
val test_bill_per_patient = test_times_visited.join(temp_test_sum_deposit,$"patientid" === $"patientid_", "left").drop("patientid_")

// converting categorically binned age to numerical
val train_age = train_bill_per_patient.withColumn("Age", regexp_replace($"Age", "0-10", "5"))
               .withColumn("Age", regexp_replace($"Age", "11-20", "15"))
               .withColumn("Age", regexp_replace($"Age", "21-30", "25"))
               .withColumn("Age", regexp_replace($"Age", "31-40", "35"))
               .withColumn("Age", regexp_replace($"Age", "41-50", "45"))
               .withColumn("Age", regexp_replace($"Age", "51-60", "55"))
               .withColumn("Age", regexp_replace($"Age", "61-70", "65"))
               .withColumn("Age", regexp_replace($"Age", "71-80", "75"))
               .withColumn("Age", regexp_replace($"Age", "81-90", "85"))
               .withColumn("Age", regexp_replace($"Age", "91-100", "95"))
               .withColumn("Age", col("Age").cast("Double"))

val test_age = test_bill_per_patient.withColumn("Age", regexp_replace($"Age", "0-10", "5"))
               .withColumn("Age", regexp_replace($"Age", "11-20", "15"))
               .withColumn("Age", regexp_replace($"Age", "21-30", "25"))
               .withColumn("Age", regexp_replace($"Age", "31-40", "35"))
               .withColumn("Age", regexp_replace($"Age", "41-50", "45"))
               .withColumn("Age", regexp_replace($"Age", "51-60", "55"))
               .withColumn("Age", regexp_replace($"Age", "61-70", "65"))
               .withColumn("Age", regexp_replace($"Age", "71-80", "75"))
               .withColumn("Age", regexp_replace($"Age", "81-90", "85"))
               .withColumn("Age", regexp_replace($"Age", "91-100", "95"))
               .withColumn("Age", col("Age").cast("Double"))

// Ordinal categories encoding
val train_ord = train_age.withColumn("Severity of Illness", regexp_replace($"Severity of Illness", "Minor", "1"))
               .withColumn("Severity of Illness", regexp_replace($"Severity of Illness", "Moderate", "2"))
               .withColumn("Severity of Illness", regexp_replace($"Severity of Illness", "Extreme", "3"))
               .withColumn("Type of Admission", regexp_replace($"Type of Admission", "Emergency", "1"))
               .withColumn("Type of Admission", regexp_replace($"Type of Admission", "Urgent", "2"))
               .withColumn("Type of Admission", regexp_replace($"Type of Admission", "Trauma", "3"))
               .withColumn("Severity of Illness", col("Severity of Illness").cast("Double"))
               .withColumn("Type of Admission", col("Type of Admission").cast("Double"))

val test_ord = test_age.withColumn("Severity of Illness", regexp_replace($"Severity of Illness", "Minor", "1"))
               .withColumn("Severity of Illness", regexp_replace($"Severity of Illness", "Moderate", "2"))
               .withColumn("Severity of Illness", regexp_replace($"Severity of Illness", "Extreme", "3"))
               .withColumn("Type of Admission", regexp_replace($"Type of Admission", "Emergency", "1"))
               .withColumn("Type of Admission", regexp_replace($"Type of Admission", "Urgent", "2"))
               .withColumn("Type of Admission", regexp_replace($"Type of Admission", "Trauma", "3"))
               .withColumn("Severity of Illness", col("Severity of Illness").cast("Double"))
               .withColumn("Type of Admission", col("Type of Admission").cast("Double"))

// creating a feature that shows number of hospitals visited for each particular patient
val tmp_train_hospitals_visited = train_ord.groupBy("patientid").agg(countDistinct("Hospital_code")).withColumnRenamed("count(Hospital_code)", "num_of_hospitals").withColumnRenamed("patientid", "patientid_")
val tmp_test_hospitals_visited = test_ord.groupBy("patientid").agg(countDistinct("Hospital_code")).withColumnRenamed("count(Hospital_code)", "num_of_hospitals").withColumnRenamed("patientid", "patientid_")

val train_num_of_hosp = train_ord.join(tmp_train_hospitals_visited,$"patientid" === $"patientid_", "left").drop("patientid_")
val test_num_of_hosp = test_ord.join(tmp_test_hospitals_visited,$"patientid" === $"patientid_", "left").drop("patientid_")

// creating a feature that shows number of departments visited for each particular patient
val tmp_train_dept_visited = train_num_of_hosp.groupBy("patientid").agg(countDistinct("Department")).withColumnRenamed("count(Department)", "num_of_departments").withColumnRenamed("patientid", "patientid_")
val tmp_test_dept_visited = test_num_of_hosp.groupBy("patientid").agg(countDistinct("Department")).withColumnRenamed("count(Department)", "num_of_departments").withColumnRenamed("patientid", "patientid_")

val train_num_of_dept = train_num_of_hosp.join(tmp_train_dept_visited,$"patientid" === $"patientid_", "left").drop("patientid_")
val test_num_of_dept = test_num_of_hosp.join(tmp_test_dept_visited,$"patientid" === $"patientid_", "left").drop("patientid_")

// boolean column to detect if the patient has changed departments?
val train_dept_changed = train_num_of_dept.withColumn("department_changed", when(col("num_of_departments") > lit(1), 1).otherwise(0))
val test_dept_changed = test_num_of_dept.withColumn("department_changed", when(col("num_of_departments") > lit(1), 1).otherwise(0))

// creating a feature that shows number of regions visited for each particular patient
val tmp_train_region_visited = train_dept_changed.groupBy("patientid").agg(countDistinct("Hospital_region_code")).withColumnRenamed("count(Hospital_region_code)", "num_of_regions").withColumnRenamed("patientid", "patientid_")
val tmp_test_region_visited = test_dept_changed.groupBy("patientid").agg(countDistinct("Hospital_region_code")).withColumnRenamed("count(Hospital_region_code)", "num_of_regions").withColumnRenamed("patientid", "patientid_")

val train_num_of_region = train_dept_changed.join(tmp_train_region_visited,$"patientid" === $"patientid_", "left").drop("patientid_")
val test_num_of_region = test_dept_changed.join(tmp_test_region_visited,$"patientid" === $"patientid_", "left").drop("patientid_")

// creating a feature that shows amount of admission types that the patient was admitted for
val tmp_train_adm_types = train_num_of_region.groupBy("patientid").agg(countDistinct("Type of Admission")).withColumnRenamed("count(Type of Admission)", "num_of_adm_types").withColumnRenamed("patientid", "patientid_")
val tmp_test_adm_types = test_num_of_region.groupBy("patientid").agg(countDistinct("Type of Admission")).withColumnRenamed("count(Type of Admission)", "num_of_adm_types").withColumnRenamed("patientid", "patientid_")

val train_engineered = train_num_of_region.join(tmp_train_adm_types,$"patientid" === $"patientid_", "left").drop("patientid_")
val test_engineered = test_num_of_region.join(tmp_test_adm_types,$"patientid" === $"patientid_", "left").drop("patientid_")

//one-hot encoding of nominal categorical features
val nomStrCol = Array("Hospital_type_code", "Hospital_region_code", "Department", "Ward_Type", "Ward_Facility_Code", "City_Code_Hospital", "Hospital_code", "City_Code_Patient")
                  
val nomIndexer = new StringIndexer().setInputCols(nomStrCol).setOutputCols(nomStrCol.map(name => s"${name}_ind"))
val encoder = new OneHotEncoder().setInputCols(nomStrCol.map(name => s"${name}_ind")).setOutputCols(nomStrCol.map(name => s"${name}_vec")).setDropLast(false)
val sparsetoDense = udf((v:Vector) => v.toDense)
val encoder_pipeline = new Pipeline().setStages(Array(nomIndexer, encoder)).fit(train_engineered)

val df_encoded_train = encoder_pipeline.transform(train_engineered)
                    .withColumn(s"${nomStrCol(0)}_vec", sparsetoDense($"${nomStrCol(0)}_vec"))
                    .withColumn(s"${nomStrCol(1)}_vec", sparsetoDense($"${nomStrCol(1)}_vec"))
                    .withColumn(s"${nomStrCol(2)}_vec", sparsetoDense($"${nomStrCol(2)}_vec"))
                    .withColumn(s"${nomStrCol(3)}_vec", sparsetoDense($"${nomStrCol(3)}_vec"))
                    .withColumn(s"${nomStrCol(4)}_vec", sparsetoDense($"${nomStrCol(4)}_vec"))
                    .withColumn(s"${nomStrCol(5)}_vec", sparsetoDense($"${nomStrCol(5)}_vec"))
                    .withColumn(s"${nomStrCol(6)}_vec", sparsetoDense($"${nomStrCol(6)}_vec"))
                    .withColumn(s"${nomStrCol(7)}_vec", sparsetoDense($"${nomStrCol(7)}_vec"))
                    .drop(nomStrCol.map(name => s"${name}_ind"):_*)
                    .drop(nomStrCol.map(name => s"${name}"):_*)
                    .drop("case_id")
                    .drop("patientid")

val df_encoded_test = encoder_pipeline.transform(test_engineered)
                    .withColumn(s"${nomStrCol(0)}_vec", sparsetoDense($"${nomStrCol(0)}_vec"))
                    .withColumn(s"${nomStrCol(1)}_vec", sparsetoDense($"${nomStrCol(1)}_vec"))
                    .withColumn(s"${nomStrCol(2)}_vec", sparsetoDense($"${nomStrCol(2)}_vec"))
                    .withColumn(s"${nomStrCol(3)}_vec", sparsetoDense($"${nomStrCol(3)}_vec"))
                    .withColumn(s"${nomStrCol(4)}_vec", sparsetoDense($"${nomStrCol(4)}_vec"))
                    .withColumn(s"${nomStrCol(5)}_vec", sparsetoDense($"${nomStrCol(5)}_vec"))
                    .withColumn(s"${nomStrCol(6)}_vec", sparsetoDense($"${nomStrCol(6)}_vec"))
                    .withColumn(s"${nomStrCol(7)}_vec", sparsetoDense($"${nomStrCol(7)}_vec"))
                    .drop(nomStrCol.map(name => s"${name}_ind"):_*)
                    .drop(nomStrCol.map(name => s"${name}"):_*)
                    .drop("case_id")
                    .drop("patientid")

// COMMAND ----------

|// selecting all columns except Stay (target column)
val selectColumns_left = df_encoded_train.columns.drop(8)//9
val selectColumns_right = df_encoded_train.columns.dropRight(17)//9
val selectColumns = Array.concat(selectColumns_left,selectColumns_right)

// vector assembler
val assembler = new VectorAssembler()
                .setInputCols(selectColumns)
                .setOutputCol("features")
 
val assembled_train = assembler.transform(df_encoded_train)
                      .drop(selectColumns.map(name => s"${name}"):_*)
                      .withColumn(s"features", sparsetoDense($"features"))
 
val assembled_test = assembler.transform(df_encoded_test)
                     .drop(selectColumns.map(name => s"${name}"):_*)
                     .withColumn(s"features", sparsetoDense($"features"))

// string indexer on target column and train-test split
val stringIndexer = new StringIndexer()
                    .setInputCol("Stay")
                    .setOutputCol("label")
                    .fit(assembled_train)
 
val train = stringIndexer.transform(assembled_train).drop("Stay")
val test = stringIndexer.transform(assembled_test).drop("Stay")

// COMMAND ----------

// instantiate the base classifier
val classifier = new LogisticRegression()
  .setMaxIter(10)
  .setTol(1E-6)
  .setLabelCol("label")
  .setFeaturesCol("features")


// COMMAND ----------

// instantiate the One Vs Rest Classifier.
val ovr = new OneVsRest().setClassifier(classifier)
// train the multiclass model.
val ovrModel = ovr.fit(train)

// COMMAND ----------

// We use a ParamGridBuilder to construct a grid of parameters to search over.
// 3 parameters = 12 possible combinations
val paramGrid = new ParamGridBuilder()
  .addGrid(classifier.elasticNetParam, Array(0.0,0.5,0.8))
  .addGrid(classifier.fitIntercept)
  .addGrid(classifier.regParam, Array(0.3,0.1,0.01))
  .build()

// COMMAND ----------

// Select (prediction, true label) and compute train error
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

// A CrossValidator() requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
val cv = new CrossValidator()
  .setEstimator(ovr)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)

// Run CrossValidation, and choose the best set of parameters.
val cvModel = cv.fit(train)

// COMMAND ----------

// Make predictions on test data. model is the model with combination of parameters
// that performed best
val predictions_train = cvModel.transform(train).select("label","features","prediction")
val predictions_test = cvModel.transform(test).select("label","features","prediction")

// COMMAND ----------

//Testing
// evaluate the model
val predictionsAndLabels = predictions_train.select("prediction", "label")
  .map(row => (row.getDouble(0), row.getDouble(1)))

val metrics = new MulticlassMetrics(predictionsAndLabels.rdd)

val confusionMatrix = metrics.confusionMatrix
// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Overall Statistics
val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")


// COMMAND ----------


//Test data
// evaluate the model
val predictionsAndLabels2 = predictions_test.select("prediction", "label")
  .map(row => (row.getDouble(0), row.getDouble(1)))

val metrics2 = new MulticlassMetrics(predictionsAndLabels2.rdd)

val confusionMatrix2 = metrics2.confusionMatrix
// Confusion matrix
println("Confusion matrix:")
println(metrics2.confusionMatrix)

// Overall Statistics
val accuracy2 = metrics2.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy2")

// Precision by label
val labels2 = metrics2.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics2.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics2.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics2.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics2.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics2.weightedPrecision}")
println(s"Weighted recall: ${metrics2.weightedRecall}")
println(s"Weighted F1 score: ${metrics2.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics2.weightedFalsePositiveRate}")


// COMMAND ----------

//xgb parameter setup
val xgbParam = Map("eta" -> 0.02,
      "max_depth" -> 7,
      "objective" -> "multi:softprob",
      "num_class" -> 6,
      "num_round" -> 100,
      "num_workers" -> 2,
      "allow_non_zero_for_missing" -> "true", 
      "missing" -> -999)
 
val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("label")

// applying xgb to train and test sets
val xgbClassificationModel = xgbClassifier.fit(train)

// metric calculation for xgb
val evaluator_xgb = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy")

//cross-validation
val paramGrid_xgb = new ParamGridBuilder()
                .addGrid(xgbClassifier.maxDepth, Array(4,7))
                .addGrid(xgbClassifier.eta, Array(0.08,0.1))
                .build()

val cv_xgb = new CrossValidator()
        .setEstimator(xgbClassifier)
        .setEvaluator(evaluator_xgb)
        .setEstimatorParamMaps(paramGrid_xgb)
        .setNumFolds(5)
        .setParallelism(2)

val cvModel_xgb = cv_xgb.fit(train)

// COMMAND ----------

val predictions_train_xgb = cvModel_xgb.transform(train).select("label","features","prediction")
val predictions_test_xgb = cvModel_xgb.transform(test).select("label","features","prediction")

// COMMAND ----------

cvModel_xgb.bestModel.extractParamMap().toString()

// COMMAND ----------

//Testing
// evaluate the model
val predictionsAndLabels = predictions_train_xgb.select("prediction", "label")
  .map(row => (row.getDouble(0), row.getDouble(1)))

val metrics = new MulticlassMetrics(predictionsAndLabels.rdd)

val confusionMatrix = metrics.confusionMatrix
// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Overall Statistics
val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")


// COMMAND ----------

//Test data
// evaluate the model
val predictionsAndLabels2 = predictions_test_xgb.select("prediction", "label")
  .map(row => (row.getDouble(0), row.getDouble(1)))

val metrics2 = new MulticlassMetrics(predictionsAndLabels2.rdd)

val confusionMatrix2 = metrics2.confusionMatrix
// Confusion matrix
println("Confusion matrix:")
println(metrics2.confusionMatrix)

// Overall Statistics
val accuracy2 = metrics2.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy2")

// Precision by label
val labels2 = metrics2.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics2.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics2.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics2.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics2.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics2.weightedPrecision}")
println(s"Weighted recall: ${metrics2.weightedRecall}")
println(s"Weighted F1 score: ${metrics2.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics2.weightedFalsePositiveRate}")

// COMMAND ----------

//Random Forest Model
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}


val rf = new RandomForestClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")

val paramGrid = new ParamGridBuilder().addGrid(rf.maxBins, Array(50,60,70)).addGrid(rf.maxDepth, Array(10,11,12))//.addGrid(rf.MinInstancesPerNode, Array(5,10))
.build() //for cross val

val evaluator = new MulticlassClassificationEvaluator() //for cross val
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy") 

val cv_rf = new CrossValidator() //for cross val
  .setEstimator(rf)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5) 

// Train model. This also runs the indexers.
val cvModel_rf = cv_rf.fit(train) 

val predictions_train_rf = cvModel_rf.transform(train).select("label","features","prediction")
val predictions_test_rf = cvModel_rf.transform(test).select("label","features","prediction")


// COMMAND ----------

//Train data
// evaluate the model
val predictionsAndLabels_train_rf = predictions_train_rf.select("prediction", "label")
  .map(row => (row.getDouble(0), row.getDouble(1)))

val metrics_rf_train = new MulticlassMetrics(predictionsAndLabels_train_rf.rdd)

// Overall Statistics
val accuracy_rf_train = metrics_rf_train.accuracy

println("Summary Statistics")
println(s"Accuracy = $accuracy_rf_train")

val confusionMatrix_rf_train = metrics_rf_train.confusionMatrix
// Confusion matrix
println("Confusion matrix:")
println(metrics_rf_train.confusionMatrix)

// Precision by label
val labels_rf_train = metrics_rf_train.labels
labels_rf_train.foreach { l =>
  println(s"Precision($l) = " + metrics_rf_train.precision(l))
}

// Recall by label
labels_rf_train.foreach { l =>
  println(s"Recall($l) = " + metrics_rf_train.recall(l))
}

// False positive rate by label
labels_rf_train.foreach { l =>
  println(s"FPR($l) = " + metrics_rf_train.falsePositiveRate(l))
}

// F-measure by label
labels_rf_train.foreach { l =>
  println(s"F1-Score($l) = " + metrics_rf_train.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics_rf_train.weightedPrecision}")
println(s"Weighted recall: ${metrics_rf_train.weightedRecall}")
println(s"Weighted F1 score: ${metrics_rf_train.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics_rf_train.weightedFalsePositiveRate}")


// COMMAND ----------

//Testing
// evaluate the model

val predictionsAndLabels_rf = predictions_test_rf.select("prediction", "label")
  .map(row => (row.getDouble(0), row.getDouble(1)))

val metrics_rf = new MulticlassMetrics(predictionsAndLabels_rf.rdd)

val confusionMatrix = metrics_rf.confusionMatrix
// Confusion matrix
println("Confusion matrix:")
println(metrics_rf.confusionMatrix)

// Overall Statistics
val accuracy_rf = metrics_rf.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy_rf")

// Precision by label
val labels_rf = metrics_rf.labels
labels_rf.foreach { l =>
  println(s"Precision($l) = " + metrics_rf.precision(l))
}

// Recall by label
labels_rf.foreach { l =>
  println(s"Recall($l) = " + metrics_rf.recall(l))
}

// False positive rate by label
labels_rf.foreach { l =>
  println(s"FPR($l) = " + metrics_rf.falsePositiveRate(l))
}

// F-measure by label
labels_rf.foreach { l =>
  println(s"F1-Score($l) = " + metrics_rf.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics_rf.weightedPrecision}")
println(s"Weighted recall: ${metrics_rf.weightedRecall}")
println(s"Weighted F1 score: ${metrics_rf.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics_rf.weightedFalsePositiveRate}")

// COMMAND ----------

cvModel_rf.bestModel.extractParamMap().toString()
