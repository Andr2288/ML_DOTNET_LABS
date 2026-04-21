using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MovieRecommendation;

string trainDataPath = Path.Combine(AppContext.BaseDirectory, "Data", "recommendation-ratings-train.csv");
string testDataPath = Path.Combine(AppContext.BaseDirectory, "Data", "recommendation-ratings-test.csv");

if (!File.Exists(trainDataPath) || !File.Exists(testDataPath))
{
    Console.WriteLine("Missing dataset CSV files in Data folder.");
    Console.WriteLine(trainDataPath);
    Console.WriteLine(testDataPath);
    return;
}

MLContext mlContext = new();

(IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext, trainDataPath, testDataPath);
ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);
EvaluateModel(mlContext, testDataView, model);
UseModelForSinglePrediction(mlContext, model);
SaveModel(mlContext, trainingDataView.Schema, model);

static (IDataView Training, IDataView Test) LoadData(MLContext mlContext, string trainingDataPath, string testDataPath)
{
    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(
        trainingDataPath,
        hasHeader: true,
        separatorChar: ',');

    IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(
        testDataPath,
        hasHeader: true,
        separatorChar: ',');

    return (trainingDataView, testDataView);
}

static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
{
    IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(
            outputColumnName: "userIdEncoded",
            inputColumnName: nameof(MovieRating.userId))
        .Append(mlContext.Transforms.Conversion.MapValueToKey(
            outputColumnName: "movieIdEncoded",
            inputColumnName: nameof(MovieRating.movieId)));

    var options = new MatrixFactorizationTrainer.Options
    {
        MatrixColumnIndexColumnName = "userIdEncoded",
        MatrixRowIndexColumnName = "movieIdEncoded",
        LabelColumnName = "Label",
        NumberOfIterations = 20,
        ApproximationRank = 100
    };

    IEstimator<ITransformer> trainerEstimator =
        estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

    Console.WriteLine("Training model (matrix factorization)...");
    ITransformer model = trainerEstimator.Fit(trainingDataView);
    Console.WriteLine("Training finished.");
    Console.WriteLine();

    return model;
}

static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
{
    Console.WriteLine("Evaluating on test data...");
    IDataView prediction = model.Transform(testDataView);
    RegressionMetrics metrics = mlContext.Regression.Evaluate(
        prediction,
        labelColumnName: "Label",
        scoreColumnName: "Score");

    Console.WriteLine($"  Root mean squared error: {metrics.RootMeanSquaredError}");
    Console.WriteLine($"  R-squared:               {metrics.RSquared}");
    Console.WriteLine();
}

static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
{
    Console.WriteLine("Single prediction (userId=6, movieId=10)...");

    PredictionEngine<MovieRating, MovieRatingPrediction> predictionEngine =
        mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

    var testInput = new MovieRating { userId = 6, movieId = 10 };
    MovieRatingPrediction movieRatingPrediction = predictionEngine.Predict(testInput);

    if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
    {
        Console.WriteLine($"Movie {testInput.movieId} is recommended for user {testInput.userId} (predicted rating {Math.Round(movieRatingPrediction.Score, 1)}).");
    }
    else
    {
        Console.WriteLine($"Movie {testInput.movieId} is not recommended for user {testInput.userId} (predicted rating {Math.Round(movieRatingPrediction.Score, 1)}).");
    }

    Console.WriteLine();
}

static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    string modelPath = Path.Combine(AppContext.BaseDirectory, "Data", "MovieRecommenderModel.zip");
    Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

    Console.WriteLine($"Saving model to {modelPath}...");
    mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
    Console.WriteLine("Done.");
}
