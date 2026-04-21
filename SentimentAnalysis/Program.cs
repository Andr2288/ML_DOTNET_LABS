using Microsoft.ML;
using Microsoft.ML.Data;
using SentimentAnalysis;

string dataPath = Path.Combine(AppContext.BaseDirectory, "Data", "yelp_labelled.txt");

if (!File.Exists(dataPath))
{
    Console.WriteLine($"Dataset not found: {dataPath}");
    return;
}

MLContext mlContext = new();
Microsoft.ML.DataOperationsCatalog.TrainTestData splitDataView = LoadData(mlContext);
ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
Evaluate(mlContext, model, splitDataView.TestSet);
UseModelWithSingleItem(mlContext, model);
UseModelWithBatchItems(mlContext, model);

static Microsoft.ML.DataOperationsCatalog.TrainTestData LoadData(MLContext mlContext)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
        path: Path.Combine(AppContext.BaseDirectory, "Data", "yelp_labelled.txt"),
        hasHeader: false);

    Microsoft.ML.DataOperationsCatalog.TrainTestData splitDataView =
        mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    return splitDataView;
}

static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text.FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: "Label",
            featureColumnName: "Features"));

    Console.WriteLine("Training model...");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("Training finished.");
    Console.WriteLine();

    return model;
}

static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    IDataView predictions = model.Transform(splitTestSet);

    CalibratedBinaryClassificationMetrics metrics =
        mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");

    Console.WriteLine("Test metrics");
    Console.WriteLine($"  Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"  AUC:      {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"  F1:       {metrics.F1Score:P2}");
    Console.WriteLine();
}

static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    PredictionEngine<SentimentData, SentimentPrediction> predictionFunction =
        mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

    SentimentData sampleStatement = new() { SentimentText = "This was a very bad steak" };

    var resultPrediction = predictionFunction.Predict(sampleStatement);

    Console.WriteLine("Single sample");
    Console.WriteLine(
        $"  \"{resultPrediction.SentimentText}\" -> {(resultPrediction.Prediction ? "Positive" : "Negative")}, probability {resultPrediction.Probability:F4}");
    Console.WriteLine();
}

static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
{
    IEnumerable<SentimentData> sentiments =
    [
        new SentimentData { SentimentText = "This was a horrible meal" },
        new SentimentData { SentimentText = "I love this spaghetti." }
    ];

    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
    IDataView predictions = model.Transform(batchComments);

    IEnumerable<SentimentPrediction> predictedResults =
        mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine("Batch samples");
    foreach (SentimentPrediction prediction in predictedResults)
    {
        Console.WriteLine(
            $"  \"{prediction.SentimentText}\" -> {(prediction.Prediction ? "Positive" : "Negative")}, probability {prediction.Probability:F4}");
    }
}
