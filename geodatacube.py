import xarray as xr
from shapely.geometry import box
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt


class GeoDataCube:
    def __init__(cube, filepath):
        cube.dataset = xr.open_dataset(filepath)

    def visualize_layer(cube, variable, time_index=0, cmap="viridis"):
        if variable not in cube.dataset.variables:
            raise ValueError(f"Dimension '{dim}' not found in dataset. Available dimensions: {list(cube.dataset.dims.keys())}")
        data = cube.dataset[variable].isel(time=time_index)
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap=cmap, origin="lower")
        plt.colorbar(label=variable)
        plt.title(f"{variable} - Time Index: {time_index}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    def visualize_mean(cube, variable):
       # Visualize the mean of a given variable (es. T2M, T10M..) over time, latitude, or longitude.
        
        if variable not in cube.dataset.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset. Available variables: {list(cube.dataset.data_vars.keys())}")

        mean_data = cube.dataset[variable].mean(dim="time")  # Mean over time

        plt.figure(figsize=(8, 6))
        img = mean_data.plot(cmap="viridis", add_colorbar=False)
        plt.title(f"Mean {variable} Over Time")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(img, label=f"{variable} Mean Value")
        plt.show()

        return mean_data

    def generate_heatmap(cube, variable, time_index=0, cmap="coolwarm"):
        data = cube.dataset[variable].isel(time=time_index)
        plt.imshow(data, cmap=cmap, origin="lower")
        plt.colorbar(label=variable)
        plt.title(f"Heatmap of {variable}")
        plt.show()

    def visualize_mean(cube, variable):
       # Visualize the mean of a variable (T2M, T10M..) over the dimentions: time, latitude, or longitude.
        
        if variable not in cube.dataset.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset. Available variables: {list(cube.dataset.data_vars.keys())}")

        mean_data = cube.dataset[variable].mean(dim="time")  # Mean over time

        plt.figure(figsize=(8, 6))
        img = mean_data.plot(cmap="viridis", add_colorbar=False)
        plt.title(f"Mean {variable} Over Time")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(img, label=f"{variable} Mean Value")
        plt.show()

        return mean_data    

    def calculate_trend(cube, variable):
        time = cube.dataset.coords["time"]
        x = np.arange(len(time))
        y = cube.dataset[variable].mean(dim=["lat", "lon"])
    
    #linear regression model (was used the second degree)
        coef = np.polyfit(x, y, 2)
        trend_line = np.polyval(coef, x)
    
    #plot
        plt.figure(figsize=(8, 6))
        plt.plot(time, y, label=f"Mean {variable}", color="blue")
        plt.plot(time, trend_line, label="Trend Line", linestyle="--", color="red")
        plt.title(f"Trend of {variable} Over Time")
        plt.xlabel("Time")
        plt.ylabel(variable)
        plt.legend()
        plt.grid()
        plt.show()
        
        return coef

    def get_metadata(cube):
        #Extracts metadata such as CRS, dimensions, and the data variables
        print("Dataset Dimensions:", cube.dataset.dims)
        print("Data Variables:", list(cube.dataset.data_vars))
        print("Coordinates:", list(cube.dataset.coords))
        print("Attributes:", cube.dataset.attrs)
        return {
            "dimensions": cube.dataset.dims,
            "data_variables": list(cube.dataset.data_vars),
            "coordinates": list(cube.dataset.coords),
            "attributes": cube.dataset.attrs,
        }

    def aggregate_time(cube, variable, method="mean", freq="M"):
        ##Aggregates data over time for a specific variable while keeping spatial dimensions.
        ## Aggregation method: ('mean', 'sum', etc.)
        ##Frequency: (M: monthly, Y for yearly)
        if variable not in cube.dataset.variables:
            raise ValueError(f"Variable '{variable}' not found in dataset.")

        if method not in ["mean", "sum", "max", "min"]:
            raise ValueError("Unsupported aggregation method.")

        if method == "mean":
            aggregated = cube.dataset[variable].resample(time=freq).mean()
        elif method == "sum":
            aggregated = cube.dataset[variable].resample(time=freq).sum()
        elif method == "max":
            aggregated = cube.dataset[variable].resample(time=freq).max()
        elif method == "min":
            aggregated = cube.dataset[variable].resample(time=freq).min()

        # Plot
        plt.figure(figsize=(10, 6))
        aggregated.isel(time=0).plot(cmap="viridis", size=8, aspect=1.5)  # ✅ No separate plt.colorbar()
        plt.title(f"{variable} ({method.title()} Aggregation - {freq})\nTime: {str(aggregated.time.values[0])[:10]}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

        return aggregated

    def resample_resolution(cube, variable, factor):
        ## Resample the dataset to a coarser resolution by an integer factor.
        ##factor: The factor to downsample the spatial resolution must be an integer
        
        if variable not in cube.dataset:
            raise ValueError(f"Variable '{variable}' not found in dataset.")
        if not isinstance(factor, int) or factor <= 0:
            raise ValueError("Factor must be a positive integer.")
        resampled = cube.dataset[variable].coarsen(lat=factor, lon=factor, boundary="trim").mean()
    
        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        cube.dataset[variable].isel(time=0).plot(ax=axes[0], cmap="coolwarm")
        axes[0].set_title(f"Original {variable}")
        resampled.isel(time=0).plot(ax=axes[1], cmap="coolwarm")  # ✅ Fixed line
        axes[1].set_title(f"Resampled {variable}")
        plt.tight_layout()
        plt.show()
    
        return resampled
           
    def compute_anomalies(cube, variable, baseline=None):
        ##Computes the dataset anomalies compared to a baseline
        ##variable for which we want to compute the anomalies
        ##baseline: Optional. If "none", it calculates the mean across time

        # input variable exists?
        if variable not in cube.dataset:
            raise ValueError(f"Variable '{variable}' not found in dataset.")

        #baseline handle
        if baseline is None or isinstance(baseline, str) and baseline.lower() == "none":
            baseline = cube.dataset[variable].mean(dim="time").astype("float32")
        else:
            try:
                baseline = float(baseline)  # Convert only if it's a valid number
            except ValueError:
                raise ValueError(f"Baseline must be a number or None, but got {baseline}")

        #anomalies
        anomalies = cube.dataset[variable].astype("float32") - baseline

        #spatial dimensions
        dims = cube.dataset[variable].dims
        spatial_dims = [dim for dim in ["lat", "lon", "latitude", "longitude"] if dim in dims]

        if len(spatial_dims) < 2:
            raise ValueError(f"Expected at least two spatial dimensions, but got: {dims}")

        # Plot of time serie
        plt.figure(figsize=(10, 6))
        anomalies.mean(dim=spatial_dims).plot(label=f"{variable} Anomalies", color="red", linestyle="-", marker="o")
        plt.axhline(0, color="black", linestyle="--", linewidth=1)  # Baseline reference
        plt.title(f"Temporal Anomalies in {variable}")
        plt.xlabel("Time")
        plt.ylabel(f"{variable} Anomaly")
        plt.legend()
        plt.grid()
        plt.show()

        return anomalies

    

    def apply_mask(cube, variable, min_value, max_value):
        ##Masks the data that falls outside a given range
        ## min_value: Minimum value
        ## max_value: Maximum value
        if variable not in cube.dataset:
            raise ValueError(f"Variable '{variable}' not found in dataset.")
        masked = cube.dataset[variable].where(
            (cube.dataset[variable] >= min_value) & (cube.dataset[variable] <= max_value))
        #plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Observed data
        original_plot = cube.dataset[variable].isel(time=0).plot(ax=axes[0], cmap="coolwarm", add_colorbar=True)
        axes[0].set_title(f"Original {variable}")
        # Masked data
        masked_plot = masked.isel(time=0).plot(ax=axes[1], cmap="coolwarm", add_colorbar=True)
        axes[1].set_title(f"Masked {variable} ({min_value} - {max_value})")
        plt.tight_layout()
        plt.show()

        return masked
