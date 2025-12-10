#!/usr/bin/env python3
"""
Urban Heat Island Detection using Landsat 8 Imagery
Converts Google Earth Engine JavaScript code to Python for terminal execution
"""

import ee
import geemap
import os
from datetime import datetime

def initialize_earth_engine():
    """Initialize Earth Engine with authentication"""
    try:
        ee.Initialize()
        print("✓ Earth Engine initialized successfully")
    except Exception as e:
        print("✗ Earth Engine initialization failed")
        print(f"  Error: {e}")
        print("\nPlease authenticate with Google Cloud:")
        print("  Run: earthengine authenticate")
        exit(1)

def load_admin_boundaries():
    """Load FAO Admin Boundaries"""
    print("\n📍 Loading FAO Admin Boundaries...")
    adminRegions = ee.FeatureCollection("FAO/GAUL/2015/level1")
    return adminRegions

def get_roi(centralCoord, adminRegions):
    """Extract Region of Interest based on central coordinate"""
    print(f"\n🎯 Extracting ROI for coordinates: {centralCoord}")
    
    cityPoint = ee.Geometry.Point(centralCoord)
    
    # Filter admin layer using city point
    analysisBoundary = adminRegions.filterBounds(cityPoint).map(
        lambda feature: feature.simplify(1000)
    )
    
    return analysisBoundary, cityPoint

def get_urban_mask(analysisBoundary, startDate, endDate):
    """Extract urban pixels using Dynamic World classification"""
    print(f"\n🏙️  Loading Dynamic World urban classification...")
    
    dynamicUrban = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
        .select('label') \
        .filterDate(startDate, endDate) \
        .filterBounds(analysisBoundary) \
        .filter(ee.Filter.calendarRange(5, 9, 'month')) \
        .mode() \
        .eq(6)  # Class 6 = Built Area
    
    print("✓ Urban mask created (Class 6 - Built Area)")
    return dynamicUrban

def load_landsat_thermal(analysisBoundary, startDate, endDate):
    """Load Landsat 8 thermal band and apply scaling"""
    print(f"\n🛰️  Loading Landsat 8 thermal data ({startDate} to {endDate})...")
    
    # First, get metadata
    metadata = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .select('ST_B10') \
        .filterBounds(analysisBoundary) \
        .filterDate(startDate, endDate) \
        .filter(ee.Filter.lt('CLOUD_COVER', 10))
    
    print(f"✓ Found {metadata.size().getInfo()} Landsat 8 scenes (cloud cover < 10%)")
    
    # Apply scaling to get temperature in Kelvin
    def apply_thermal_scaling(image):
        scaleFactor = ee.Number(image.get('TEMPERATURE_MULT_BAND_ST_B10'))
        offsetFactor = ee.Number(image.get('TEMPERATURE_ADD_BAND_ST_B10'))
        brightnessTemp = image.multiply(scaleFactor).add(offsetFactor)
        return brightnessTemp.copyProperties(image, image.propertyNames())
    
    landsatThermal = metadata.map(apply_thermal_scaling)
    
    return landsatThermal

def compute_lst(landsatThermal, analysisBoundary):
    """Compute median Land Surface Temperature"""
    print("\n🌡️  Computing median Land Surface Temperature...")
    
    medianThermal = landsatThermal.median()
    
    # Calculate mean LST across ROI
    meanLST = ee.Number(
        medianThermal.reduceRegion({
            'reducer': ee.Reducer.mean(),
            'geometry': analysisBoundary,
            'scale': 100,
            'maxPixels': 1e13
        }).values().get(0)
    )
    
    meanLST_value = meanLST.getInfo()
    print(f"✓ Mean LST: {meanLST_value:.2f} K ({meanLST_value - 273.15:.2f}°C)")
    
    return medianThermal, meanLST_value

def calculate_uhi_index(medianThermal, meanLST):
    """Calculate Urban Heat Island Index"""
    print("\n📊 Calculating UHI Index...")
    
    uhiIndex = medianThermal.expression(
        '(TIR - MEAN) / MEAN',
        {
            'TIR': medianThermal,
            'MEAN': meanLST
        }
    ).rename('UHI_Index')
    
    print("✓ UHI Index calculated (relative deviation from mean)")
    return uhiIndex

def classify_uhi_intensity(uhiIndex, dynamicUrban):
    """Classify UHI into 5 intensity categories"""
    print("\n🔥 Classifying UHI intensity levels...")
    
    uhiClasses = ee.Image.constant(0) \
        .where(uhiIndex.gte(0).And(uhiIndex.lt(0.005)), 1) \
        .where(uhiIndex.gte(0.005).And(uhiIndex.lt(0.010)), 2) \
        .where(uhiIndex.gte(0.010).And(uhiIndex.lt(0.015)), 3) \
        .where(uhiIndex.gte(0.015).And(uhiIndex.lt(0.020)), 4) \
        .where(uhiIndex.gte(0.020), 5) \
        .updateMask(dynamicUrban)
    
    print("✓ Classification levels:")
    print("  1 = Mild        (0.000 - 0.005)")
    print("  2 = Moderate    (0.005 - 0.010)")
    print("  3 = Strong      (0.010 - 0.015)")
    print("  4 = Very Strong (0.015 - 0.020)")
    print("  5 = Extreme     (≥ 0.020)")
    
    return uhiClasses

def export_to_drive(uhiClasses, analysisBoundary, output_folder='UrbanHeat'):
    """Export UHI classified layer to Google Drive"""
    print(f"\n💾 Exporting to Google Drive ({output_folder}/)...")
    
    task = ee.batch.Export.image.toDrive(
        image=uhiClasses.clip(analysisBoundary),
        description='UHI_Classes_Landsat8_Export',
        folder=output_folder,
        fileNamePrefix='uhi_classes',
        region=analysisBoundary,
        scale=100,
        crs='EPSG:4326',
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    
    task.start()
    print(f"✓ Export task started: {task.id}")
    print(f"  Check your Google Drive ({output_folder}/) for results")
    print(f"  Task ID: {task.id}")
    
    return task

def visualize_interactive(medianThermal, uhiClasses, analysisBoundary, centralCoord):
    """Create interactive map with geemap"""
    print("\n🗺️  Creating interactive visualization...")
    
    try:
        Map = geemap.Map(center=(centralCoord[1], centralCoord[0]), zoom=10)
        
        # Add layers
        Map.addLayer(
            medianThermal.clip(analysisBoundary),
            {'min': 280, 'max': 310, 'palette': ['blue', 'cyan', 'green', 'yellow', 'red']},
            'LST (Kelvin)'
        )
        
        Map.addLayer(
            uhiClasses.clip(analysisBoundary),
            {'min': 1, 'max': 5, 'palette': ['white', 'yellow', 'orange', 'red', 'darkred']},
            'UHI Classes'
        )
        
        Map.addLayerControl()
        
        # Save to HTML
        output_file = 'uhi_map.html'
        Map.save(output_file)
        print(f"✓ Interactive map saved: {output_file}")
        print(f"  Open this file in your browser to view the visualization")
        
    except Exception as e:
        print(f"⚠️  Could not create interactive map: {e}")
        print("  (geemap may not be installed. Install with: pip install geemap)")

def main():
    """Main execution function"""
    print("=" * 60)
    print("Urban Heat Island Detection using Landsat 8")
    print("Google Earth Engine Python API")
    print("=" * 60)
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Configuration
    centralCoord = [80.2707, 13.0827]  # Chennai, India
    startDate = '2023-01-01'
    endDate = '2024-12-31'
    
    print(f"\n⚙️  Configuration:")
    print(f"  Location: {centralCoord}")
    print(f"  Analysis Period: {startDate} to {endDate}")
    print(f"  Focus Months: May - September (summer)")
    
    # Step 1: Load administrative boundaries
    adminRegions = load_admin_boundaries()
    
    # Step 2: Extract ROI
    analysisBoundary, cityPoint = get_roi(centralCoord, adminRegions)
    
    # Step 3: Get urban mask
    dynamicUrban = get_urban_mask(analysisBoundary, startDate, endDate)
    
    # Step 4: Load Landsat 8 thermal data
    landsatThermal = load_landsat_thermal(analysisBoundary, startDate, endDate)
    
    # Step 5: Compute LST
    medianThermal, meanLST = compute_lst(landsatThermal, analysisBoundary)
    
    # Step 6: Calculate UHI Index
    uhiIndex = calculate_uhi_index(medianThermal, meanLST)
    
    # Step 7: Classify UHI intensity
    uhiClasses = classify_uhi_intensity(uhiIndex, dynamicUrban)
    
    # Step 8: Export to Google Drive
    task = export_to_drive(uhiClasses, analysisBoundary)
    
    # Step 9: Create interactive visualization (optional)
    visualize_interactive(medianThermal, uhiClasses, analysisBoundary, centralCoord)
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Analysis Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Check your Google Drive for the exported GeoTIFF")
    print("  2. Open uhi_map.html in your browser for visualization")
    print("  3. Use QGIS or ArcGIS to further analyze the results")
    print("\nFor real-time task status, visit:")
    print("  https://code.earthengine.google.com/tasks")
    print("=" * 60)

if __name__ == "__main__":
    main()
