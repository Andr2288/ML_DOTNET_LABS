using Microsoft.ML;
using Microsoft.ML.Data;
using GitHubIssueClassification;

string trainDataPath = Path.Combine(AppContext.BaseDirectory, "Data", "issues_train.tsv");
string testDataPath = Path.Combine(AppContext.BaseDirectory, "Data", "issues_test.tsv");
string modelPath = Path.Combine(AppContext.BaseDirectory, "Models", "model.zip");

if (!File.Exists(trainDataPath) || !File.Exists(testDataPath))
{
    Console.WriteLine("Dataset files are missing.");
    Console.WriteLine(trainDataPath);
    Console.WriteLine(testDataPath);
    return;
}

Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

MLContext mlContext = new(seed: 0);
IDataView trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(trainDataPath, hasHeader: true);

IEstimator<ITransformer> preprocessing = ProcessData(mlContext);
ITransformer trainedModel = BuildAndTrainModel(mlContext, trainingDataView, preprocessing);
Evaluate(mlContext, trainingDataView.Schema, testDataPath, trainedModel, modelPath);
PredictIssue(mlContext, modelPath);

static IEstimator<ITransformer> ProcessData(MLContext mlContext)
{
    return mlContext.Transforms.Conversion.MapValueToKey(
            inputColumnName: nameof(GitHubIssue.Area),
            outputColumnName: "Label")
        .Append(mlContext.Transforms.Text.FeaturizeText(
            outputColumnName: "TitleFeaturized",
            inputColumnName: nameof(GitHubIssue.Title)))
        .Append(mlContext.Transforms.Text.FeaturizeText(
            outputColumnName: "DescriptionFeaturized",
            inputColumnName: nameof(GitHubIssue.Description)))
        .Append(mlContext.Transforms.Concatenate(
            "Features",
            "TitleFeaturized",
            "DescriptionFeaturized"))
        .AppendCacheCheckpoint(mlContext);
}

static ITransformer BuildAndTrainModel(
    MLContext mlContext,
    IDataView trainingDataView,
    IEstimator<ITransformer> pipeline)
{
    IEstimator<ITransformer> trainingPipeline = pipeline
        .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
            labelColumnName: "Label",
            featureColumnName: "Features"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel"));

    ITransformer model = trainingPipeline.Fit(trainingDataView);

    PredictionEngine<GitHubIssue, IssuePrediction> predEngine =
        mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(model);

    GitHubIssue issue = new()
    {
        Title = "WebSockets communication is slow in my machine",
        Description =
            "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
    };

    IssuePrediction prediction = predEngine.Predict(issue);
    Console.WriteLine($"Prediction (fresh model): {prediction.Area}");
    Console.WriteLine();

    return model;
}

static void Evaluate(
    MLContext mlContext,
    DataViewSchema trainingDataViewSchema,
    string testDataPath,
    ITransformer trainedModel,
    string modelPath)
{
    IDataView testDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(testDataPath, hasHeader: true);

    MulticlassClassificationMetrics testMetrics = mlContext.MulticlassClassification.Evaluate(
        trainedModel.Transform(testDataView),
        labelColumnName: "Label");

    Console.WriteLine("Test metrics");
    Console.WriteLine($"  MicroAccuracy:     {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"  MacroAccuracy:     {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"  LogLoss:           {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"  LogLossReduction:  {testMetrics.LogLossReduction:0.###}");
    Console.WriteLine();

    mlContext.Model.Save(trainedModel, trainingDataViewSchema, modelPath);
    Console.WriteLine($"Model saved: {modelPath}");
    Console.WriteLine();
}

static void PredictIssue(MLContext mlContext, string modelPath)
{
    ITransformer loadedModel = mlContext.Model.Load(modelPath, out _);

    PredictionEngine<GitHubIssue, IssuePrediction> predEngine =
        mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

    GitHubIssue singleIssue = new()
    {
        Title = "Entity Framework crashes",
        Description = "When connecting to the database, EF is crashing"
    };

    IssuePrediction prediction = predEngine.Predict(singleIssue);
    Console.WriteLine($"Prediction (loaded model): {prediction.Area}");
}
