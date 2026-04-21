using Microsoft.ML;
using ProductSalesAnomalyDetection;

const int DocSize = 36;

string dataPath = Path.Combine(AppContext.BaseDirectory, "Data", "product-sales.csv");

if (!File.Exists(dataPath))
{
    Console.WriteLine($"Dataset not found: {dataPath}");
    return;
}

MLContext mlContext = new();
IDataView dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(
    path: dataPath,
    hasHeader: true,
    separatorChar: ',');

DetectSpike(mlContext, DocSize, dataView);
DetectChangepoint(mlContext, DocSize, dataView);

static IDataView CreateEmptyDataView(MLContext mlContext)
{
    IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
    return mlContext.Data.LoadFromEnumerable(enumerableData);
}

static void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
{
    Console.WriteLine("Spike detection (temporary bursts)");
    Console.WriteLine("Training transform...");

    var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(
        outputColumnName: nameof(SpikePredictionRow.Prediction),
        inputColumnName: nameof(ProductSalesData.numSales),
        confidence: 95d,
        pvalueHistoryLength: docSize / 4);

    ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));
    Console.WriteLine("Done.");
    Console.WriteLine();

    IDataView transformedData = iidSpikeTransform.Transform(productSales);

    var predictions = mlContext.Data.CreateEnumerable<SpikePredictionRow>(transformedData, reuseRowObject: false);

    Console.WriteLine("Alert\tScore\tP-Value");
    foreach (SpikePredictionRow p in predictions)
    {
        if (p.Prediction is null)
        {
            continue;
        }

        var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";
        if (p.Prediction[0] == 1)
        {
            results += " <-- Spike detected";
        }

        Console.WriteLine(results);
    }

    Console.WriteLine();
}

static void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
{
    Console.WriteLine("Change point detection (persistent shifts)");
    Console.WriteLine("Training transform...");

    var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(
        outputColumnName: nameof(ChangePointPredictionRow.Prediction),
        inputColumnName: nameof(ProductSalesData.numSales),
        confidence: 95d,
        changeHistoryLength: docSize / 4);

    ITransformer iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));
    Console.WriteLine("Done.");
    Console.WriteLine();

    IDataView transformedData = iidChangePointTransform.Transform(productSales);

    var predictions =
        mlContext.Data.CreateEnumerable<ChangePointPredictionRow>(transformedData, reuseRowObject: false);

    Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");
    foreach (ChangePointPredictionRow p in predictions)
    {
        if (p.Prediction is null)
        {
            continue;
        }

        var results =
            $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";
        if (p.Prediction[0] == 1)
        {
            results += " <-- alert is on, predicted changepoint";
        }

        Console.WriteLine(results);
    }

    Console.WriteLine();
}
