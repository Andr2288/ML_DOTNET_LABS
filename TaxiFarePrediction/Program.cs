using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace TaxiFarePrediction;

internal static class Program
{
    private static readonly string TrainDataPath = Path.Combine(
        Environment.CurrentDirectory,
        "Data",
        "taxi-fare-train.csv");

    private static readonly string TestDataPath = Path.Combine(
        Environment.CurrentDirectory,
        "Data",
        "taxi-fare-test.csv");

    private static void Main()
    {
        MLContext mlContext = new(seed: 0);
        ITransformer model = Train(mlContext, TrainDataPath);
        Evaluate(mlContext, model);
        TestSinglePrediction(mlContext, model);
    }

    private static ITransformer Train(MLContext mlContext, string dataPath)
    {
        IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(
            dataPath,
            hasHeader: true,
            separatorChar: ',');

        var pipeline = mlContext.Transforms
            .CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "VendorIdEncoded",
                inputColumnName: "VendorId"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "RateCodeEncoded",
                inputColumnName: "RateCode"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "PaymentTypeEncoded",
                inputColumnName: "PaymentType"))
            .Append(mlContext.Transforms.Concatenate(
                "Features",
                "VendorIdEncoded",
                "RateCodeEncoded",
                "PassengerCount",
                "TripDistance",
                "PaymentTypeEncoded"))
            .Append(mlContext.Regression.Trainers.FastTree());

        return pipeline.Fit(dataView);
    }

    private static void Evaluate(MLContext mlContext, ITransformer model)
    {
        IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(
            TestDataPath,
            hasHeader: true,
            separatorChar: ',');

        IDataView predictions = model.Transform(dataView);
        RegressionMetrics metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

        Console.WriteLine("Model quality (test set)");
        Console.WriteLine($"  RSquared: {metrics.RSquared:0.##}");
        Console.WriteLine($"  RMSE:     {metrics.RootMeanSquaredError:0.##}");
    }

    private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
    {
        PredictionEngine<TaxiTrip, TaxiTripFarePrediction> predictionFunction =
            mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

        var taxiTripSample = new TaxiTrip
        {
            VendorId = "VTS",
            RateCode = "1",
            PassengerCount = 1,
            TripTime = 1140,
            TripDistance = 3.75f,
            PaymentType = "CRD",
            FareAmount = 0,
        };

        TaxiTripFarePrediction prediction = predictionFunction.Predict(taxiTripSample);

        Console.WriteLine();
        Console.WriteLine("Sample prediction");
        Console.WriteLine($"  Predicted fare: {prediction.FareAmount:0.####}");
        Console.WriteLine($"  Actual fare:    15.5");
    }
}
