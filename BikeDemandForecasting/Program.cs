using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

string rootDir =
    Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));
string dataPath = Path.Combine(rootDir, "Data", "DailyDemand.json");
string modelPath = Path.Combine(rootDir, "MLModel.zip");

MLContext mlContext = new MLContext();

List<ModelInput> rows = LoadRentalsFromJson(dataPath);
IDataView dataView = mlContext.Data.LoadFromEnumerable(rows);

IDataView firstYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", upperBound: 1);
IDataView secondYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", lowerBound: 1);

var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: "ForecastedRentals",
    inputColumnName: "TotalRentals",
    windowSize: 7,
    seriesLength: 30,
    trainSize: 365,
    horizon: 7,
    confidenceLevel: 0.95f,
    confidenceLowerBoundColumn: "LowerBoundRentals",
    confidenceUpperBoundColumn: "UpperBoundRentals");

SsaForecastingTransformer forecaster = forecastingPipeline.Fit(firstYearData);

Evaluate(secondYearData, forecaster, mlContext);

var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
forecastEngine.CheckPoint(mlContext, modelPath);

Forecast(secondYearData, 7, forecastEngine, mlContext);

static List<ModelInput> LoadRentalsFromJson(string jsonPath)
{
    string json = File.ReadAllText(jsonPath);
    var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
    var list = JsonSerializer.Deserialize<List<ModelInput>>(json, options)
               ?? throw new InvalidOperationException($"Failed to deserialize {jsonPath}");
    return list;
}

void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
{
    IDataView predictions = model.Transform(testData);

    IEnumerable<float> actual =
        mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
            .Select(observed => observed.TotalRentals);

    IEnumerable<float> forecast =
        mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
            .Select(prediction => prediction.ForecastedRentals[0]);

    var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

    var MAE = metrics.Average(error => Math.Abs(error));
    var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2)));

    Console.WriteLine("Evaluation Metrics");
    Console.WriteLine("---------------------");
    Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
    Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
}

void Forecast(
    IDataView testData,
    int horizon,
    TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster,
    MLContext mlContext)
{
    ModelOutput forecast = forecaster.Predict();

    IEnumerable<string> forecastOutput =
        mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
            .Take(horizon)
            .Select((ModelInput rental, int index) =>
            {
                string rentalDate = rental.RentalDate.ToShortDateString();
                float actualRentals = rental.TotalRentals;
                float lowerEstimate = Math.Max(0, forecast.LowerBoundRentals[index]);
                float estimate = forecast.ForecastedRentals[index];
                float upperEstimate = forecast.UpperBoundRentals[index];
                return $"Date: {rentalDate}\n" +
                       $"Actual Rentals: {actualRentals}\n" +
                       $"Lower Estimate: {lowerEstimate}\n" +
                       $"Forecast: {estimate}\n" +
                       $"Upper Estimate: {upperEstimate}\n";
            });

    Console.WriteLine("Rental Forecast");
    Console.WriteLine("---------------------");
    foreach (var prediction in forecastOutput)
    {
        Console.WriteLine(prediction);
    }
}

public class ModelInput
{
    public DateTime RentalDate { get; set; }
    public float Year { get; set; }
    public float TotalRentals { get; set; }
}

public class ModelOutput
{
    public float[] ForecastedRentals { get; set; } = null!;
    public float[] LowerBoundRentals { get; set; } = null!;
    public float[] UpperBoundRentals { get; set; } = null!;
}
