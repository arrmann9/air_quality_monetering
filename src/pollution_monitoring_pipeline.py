import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import h5py
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.morphology import binary_opening, disk
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
import joblib

# Create output directory structure
os.makedirs("outputs/preprocessed", exist_ok=True)
os.makedirs("outputs/cloud_masks", exist_ok=True)
os.makedirs("outputs/aligned_data", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/pm_maps", exist_ok=True)
os.makedirs("outputs/visualizations", exist_ok=True)

# Constants
SOLAR_IRRADIANCE = 1362.0  # W/m²/μm

def save_output(data, filename, description):
    """Save output with metadata tracking"""
    np.save(filename, data)
    print(f"Saved {description}: {os.path.basename(filename)}")

def save_dataframe(df, filename, description):
    """Save DataFrame with metadata tracking"""
    df.to_csv(filename, index=False)
    print(f"Saved {description}: {os.path.basename(filename)}")

def save_plot(fig, filename, description):
    """Save visualization with metadata tracking"""
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved {description}: {os.path.basename(filename)}")

def preprocess_insat_data(h5_path, output_dir="outputs/preprocessed"):
    """Step 1: Preprocess INSAT-3D data"""
    with h5py.File(h5_path, 'r') as f:
        radiance = f['VIS_RADIANCE'][0]
        sun_elevation = f['Sun_Elevation'][0]
        time_val = f['time'][0]

    # Convert timestamp
    acq_time = datetime(1970, 1, 1) + timedelta(seconds=float(time_val))
    print(f"\nAcquisition Time: {acq_time}")

    # Correct sun elevation
    sun_elevation = sun_elevation.astype(np.float32)
    invalid_mask = np.abs(sun_elevation) > 1000
    if np.nanmax(sun_elevation) > 360:
        sun_elevation /= 10.0
    sun_elevation = np.clip(sun_elevation, -90, 90)
    sun_elevation[invalid_mask] = np.nan

    # Convert to reflectance
    day_of_year = acq_time.timetuple().tm_yday
    d_au = 1 - 0.01672 * np.cos(np.radians(0.9856 * (day_of_year - 4)))
    sza = 90 - sun_elevation
    reflectance = (np.pi * radiance * d_au**2) / (SOLAR_IRRADIANCE * np.cos(np.radians(sza)))
    reflectance[sun_elevation < 5] = np.nan
    reflectance[(reflectance < 0) | (reflectance > 1)] = np.nan

    # Save outputs
    save_output(reflectance, f"{output_dir}/insat_reflectance.npy", "reflectance data")
    save_output(sun_elevation, f"{output_dir}/insat_sun_elevation.npy", "sun elevation data")
    
    # Save metadata
    metadata = {
        'acquisition_time': str(acq_time),
        'data_shape': str(reflectance.shape),
        'reflectance_range': f"{np.nanmin(reflectance):.4f}-{np.nanmax(reflectance):.4f}",
        'sun_elevation_range': f"{np.nanmin(sun_elevation):.1f}-{np.nanmax(sun_elevation):.1f}"
    }
    pd.DataFrame([metadata]).to_csv(f"{output_dir}/metadata.csv", index=False)
    
    return reflectance, sun_elevation, acq_time

def generate_cloud_mask(reflectance, output_dir="outputs/cloud_masks"):
    """Step 2: Generate cloud mask"""
    valid_mask = ~np.isnan(reflectance)
    cloud_mask = np.zeros(reflectance.shape, dtype=np.uint8)
    
    if np.any(valid_mask):
        refl_values = reflectance[valid_mask].flatten()
        p75 = np.percentile(refl_values, 75)
        bright_mask = reflectance > p75
        smooth_refl = gaussian_filter(reflectance, sigma=5)
        texture = np.abs(reflectance - smooth_refl)
        homogeneous_bright = bright_mask & (texture < 0.01)
        cloud_mask[homogeneous_bright & valid_mask] = 1
        cloud_mask = binary_dilation(cloud_mask, structure=disk(3))
        cloud_mask = binary_opening(cloud_mask, disk(3)).astype(np.uint8)
    
    # Save outputs
    save_output(cloud_mask, f"{output_dir}/cloud_mask.npy", "cloud mask")
    
    # Calculate statistics
    valid_pixels = np.sum(valid_mask)
    cloud_pixels = np.sum(cloud_mask[valid_mask])
    stats = {
        'cloud_coverage_percent': (cloud_pixels / valid_pixels) * 100 if valid_pixels > 0 else 0,
        'cloud_pixels': int(cloud_pixels),
        'clear_sky_pixels': int(valid_pixels - cloud_pixels)
    }
    pd.DataFrame([stats]).to_csv(f"{output_dir}/cloud_stats.csv", index=False)
    
    return cloud_mask

def align_satellite_ground_data(reflectance, sun_elevation, cpcb_path, output_dir="outputs/aligned_data"):
    """Step 3: Align satellite and ground data"""
    cpcb = pd.read_csv(cpcb_path)
    
    # Ensure required columns exist
    required_cols = ['station_id', 'latitude', 'longitude', 'Timestamp', 
                    'PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'RH (%)', 'AT (°C)']
    if not all(col in cpcb.columns for col in required_cols):
        missing = set(required_cols) - set(cpcb.columns)
        raise ValueError(f"Missing columns in CPCB data: {missing}")
    
    # Process CPCB data
    cpcb = cpcb.rename(columns={
        'station_id': 'StationID',
        'latitude': 'Latitude',
        'longitude': 'Longitude'
    })
    cpcb['Timestamp'] = pd.to_datetime(cpcb['Timestamp'])
    daily_cpcb = cpcb.groupby(['StationID', 'Latitude', 'Longitude']).agg({
        'PM2.5 (µg/m³)': 'mean',
        'PM10 (µg/m³)': 'mean',
        'RH (%)': 'mean',
        'AT (°C)': 'mean'
    }).reset_index()
    
    # Create grid for India
    lat_grid = np.linspace(37, 8, reflectance.shape[0])
    lon_grid = np.linspace(68, 97, reflectance.shape[1])
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Prepare satellite data
    valid_mask = ~np.isnan(reflectance)
    valid_indices = np.where(valid_mask)
    valid_lons = lon_mesh[valid_indices]
    valid_lats = lat_mesh[valid_indices]
    valid_reflectance = reflectance[valid_indices]
    valid_sun_elev = sun_elevation[valid_indices]
    
    # Align data
    sat_tree = cKDTree(np.column_stack([valid_lons, valid_lats]))
    station_coords = daily_cpcb[['Longitude', 'Latitude']].values
    distances, indices = sat_tree.query(station_coords, k=1)
    
    aligned_data = []
    for i, row in daily_cpcb.iterrows():
        sat_idx = indices[i]
        aligned_data.append({
            'StationID': row['StationID'],
            'Ground_Lat': row['Latitude'],
            'Ground_Lon': row['Longitude'],
            'Sat_Lat': valid_lats[sat_idx],
            'Sat_Lon': valid_lons[sat_idx],
            'Distance_km': distances[i] * 111,
            'PM2.5': row['PM2.5 (µg/m³)'],
            'PM10': row['PM10 (µg/m³)'],
            'Reflectance': valid_reflectance[sat_idx],
            'Sun_Elevation': valid_sun_elev[sat_idx],
            'RH': row['RH (%)'],
            'Temp': row['AT (°C)']
        })
    
    aligned_df = pd.DataFrame(aligned_data)
    
    # Save outputs
    save_dataframe(aligned_df, f"{output_dir}/aligned_data.csv", "aligned satellite-ground data")
    
    # Visualization
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(lon_mesh, lat_mesh, c=reflectance, s=1, alpha=0.3, cmap='viridis', vmin=0, vmax=0.1)
    plt.scatter(aligned_df['Ground_Lon'], aligned_df['Ground_Lat'], c='red', s=50, label='Stations')
    plt.colorbar(label='Reflectance')
    plt.title('Satellite Data & Air Quality Stations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(68, 97)
    plt.ylim(8, 37)
    plt.legend()
    save_plot(fig, f"{output_dir}/station_alignment.png", "station alignment visualization")
    
    return aligned_df

def train_pm_models(aligned_df, output_dir="outputs/models"):
    """Step 4: Train PM estimation models"""
    # Feature engineering
    aligned_df['Reflectance_Squared'] = aligned_df['Reflectance'] ** 2
    aligned_df['RH_Temp_Interaction'] = aligned_df['RH'] * aligned_df['Temp']
    
    # Prepare data
    features = aligned_df[['Reflectance', 'Sun_Elevation', 'RH', 'Temp', 
                          'Reflectance_Squared', 'RH_Temp_Interaction']]
    target_pm25 = aligned_df['PM2.5']
    target_pm10 = aligned_df['PM10']
    
    # Train models
    pm25_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    pm10_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    pm25_model.fit(features, target_pm25)
    pm10_model.fit(features, target_pm10)
    
    # Save models
    joblib.dump(pm25_model, f"{output_dir}/pm25_model.pkl")
    joblib.dump(pm10_model, f"{output_dir}/pm10_model.pkl")
    print(f"Saved models to {output_dir}")
    
    return pm25_model, pm10_model

def generate_pm_maps(reflectance, sun_elevation, cloud_mask, pm25_model, pm10_model, output_dir="outputs/pm_maps"):
    """Step 5: Generate PM concentration maps"""
    # Apply cloud mask
    reflectance[cloud_mask == 1] = np.nan
    
    # Prepare feature grid
    grid_size = reflectance.size
    grid_features = pd.DataFrame({
        'Reflectance': reflectance.flatten(),
        'Sun_Elevation': sun_elevation.flatten(),
        'RH': np.full(grid_size, 50),  # Placeholder
        'Temp': np.full(grid_size, 25)  # Placeholder
    })
    grid_features['Reflectance_Squared'] = grid_features['Reflectance'] ** 2
    grid_features['RH_Temp_Interaction'] = grid_features['RH'] * grid_features['Temp']
    
    # Predict PM concentrations
    pm25_map = pm25_model.predict(grid_features).reshape(reflectance.shape)
    pm10_map = pm10_model.predict(grid_features).reshape(reflectance.shape)
    
    # Save outputs
    save_output(pm25_map, f"{output_dir}/pm25_map.npy", "PM2.5 concentration map")
    save_output(pm10_map, f"{output_dir}/pm10_map.npy", "PM10 concentration map")
    
    # Generate visualizations
    def create_pm_plot(data, title, pollutant, filename):
        fig = plt.figure(figsize=(12, 10))
        plt.imshow(data, cmap='RdYlGn_r', vmin=0, vmax=100 if pollutant == 'PM2.5' else 200, 
                  aspect='auto', extent=[68, 97, 8, 37])
        cbar = plt.colorbar(shrink=0.7)
        cbar.set_label('µg/m³', rotation=270, labelpad=20)
        plt.title(title, fontsize=16)
        plt.xlabel('Longitude', fontsize=14)
        plt.ylabel('Latitude', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.plot([68, 97, 97, 68, 68], [8, 8, 37, 37, 8], 'k-', lw=1)
        save_plot(fig, filename, f"{pollutant} map")
    
    create_pm_plot(pm25_map, 'India PM2.5 Concentration', 'PM2.5', 
                  f"{output_dir}/pm25_map.png")
    create_pm_plot(pm10_map, 'India PM10 Concentration', 'PM10', 
                  f"{output_dir}/pm10_map.png")
    
    return pm25_map, pm10_map

def main():
    """Main workflow execution"""
    # Configuration - update these paths as needed
    INSAT_H5_PATH = "3DIMG_18JUN2024_0000_L1C_ASIA_MER_V01R00_B1-1.h5"
    CPCB_CSV_PATH = "processed_cpcb_data.csv"
    
    try:
        # Step 1: Preprocess satellite data
        reflectance, sun_elevation, acq_time = preprocess_insat_data(INSAT_H5_PATH)
        
        # Step 2: Generate cloud mask
        cloud_mask = generate_cloud_mask(reflectance)
        
        # Step 3: Align satellite and ground data
        aligned_df = align_satellite_ground_data(reflectance, sun_elevation, CPCB_CSV_PATH)
        
        # Step 4: Train PM models
        pm25_model, pm10_model = train_pm_models(aligned_df)
        
        # Step 5: Generate PM maps
        pm25_map, pm10_map = generate_pm_maps(reflectance, sun_elevation, cloud_mask, 
                                             pm25_model, pm10_model)
        
        print("\nPipeline executed successfully! Outputs saved in 'outputs' directory.")
        print("Next steps: Integrate with backend using files in 'outputs' directory.")
        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        print("Check input files and directory permissions")

if __name__ == "__main__":
    main()