using Microsoft.ML.Data;

namespace ProductSalesAnomalyDetection;

public class ProductSalesData
{
    [LoadColumn(0)]
    public string? Month;

    [LoadColumn(1)]
    public float numSales;
}

/// <summary>Output of <see cref="Microsoft.ML.TransformsCatalog.DetectIidSpike"/> (alert, score, p-value).</summary>
public class SpikePredictionRow
{
    [VectorType(3)]
    public double[]? Prediction { get; set; }
}

/// <summary>Output of <see cref="Microsoft.ML.TransformsCatalog.DetectIidChangePoint"/> (+ martingale).</summary>
public class ChangePointPredictionRow
{
    [VectorType(4)]
    public double[]? Prediction { get; set; }
}
