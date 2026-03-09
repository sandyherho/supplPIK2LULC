#!/usr/bin/env python
"""
PIK2 Coastal and Regional Geospatial Context Map
Generates a high-resolution 2-panel map (Indonesia & PIK2 Area) using PyGMT.
Extracts and reports bathymetry and elevation statistics for the PIK2 region.

Author : Sandy H. S. Herho
Date   : 2026/03/10
License: MIT
"""

import os
import sys
import numpy as np
import pygmt

class PIK2MapGenerator:
    def __init__(self):
        # 1. Output Directories
        self.figs_dir = "../figs"
        self.reports_dir = "../reports"
        self._setup_directories()

        # 2. Coordinates & Bounding Boxes
        # Panel A: Indonesia General (Keep same as original request)
        self.reg_indo = [90.0, 145.0, -13.0, 8.0]               
        
        # Panel B: PIK2 Region
        # lon 106.63–106.77, lat -6.08–-5.98
        self.reg_study = [106.63, 106.77, -6.08, -5.98]         
        
        # Center point for a marker (PIK2 Landmark area)
        self.pik2_lon = 106.70
        self.pik2_lat = -6.03

        # 3. Data Grids Placeholder
        self.grid_study = None

    def _setup_directories(self):
        """Create necessary output directories if they don't exist."""
        for directory in [self.figs_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory verified: {directory}")

    def load_grids_safely(self):
        """Loads grids with error catching for corrupted network cache."""
        try:
            print(f"Loading high-res grid for PIK2 area: {self.reg_study}")
            # Use 01s or 15s depending on detail needed; 15s is standard for regional.
            self.grid_study = pygmt.datasets.load_earth_relief(
                resolution="15s", 
                region=self.reg_study
            )
        except ValueError:
            print("\n[CRITICAL ERROR] Corrupted GMT Cache Detected.")
            print("Run: rm -rf ~/.gmt/server/earth/earth_relief\n")
            sys.exit(1)

    def generate_bathymetry_report(self):
        """Extract bathymetry/topography data and generate a statistical report."""
        values = self.grid_study.values
        ocean_mask = values < 0
        land_mask = values >= 0
        
        bathymetry = values[ocean_mask]
        elevation = values[land_mask]

        report_path = os.path.join(self.reports_dir, "PIK2_Geomorphology_Report.txt")
        
        with open(report_path, "w") as f:
            f.write("=================================================================\n")
            f.write(" GEOMORPHOLOGICAL REPORT: PIK2 COASTAL AREA\n")
            f.write(f" Bounding Box : Lon {self.reg_study[0:2]}, Lat {self.reg_study[2:4]}\n")
            f.write(" Dataset Resolution   : 15 arc-seconds\n")
            f.write("=================================================================\n\n")
            
            if len(bathymetry) > 0:
                f.write("[ MARINE BATHYMETRY (JAVA SEA) ]\n")
                f.write(f"Maximum Depth        : {np.min(bathymetry):.2f} meters\n")
                f.write(f"Average Depth (Mean): {np.mean(bathymetry):.2f} meters\n\n")
            
            if len(elevation) > 0:
                f.write("[ TERRESTRIAL TOPOGRAPHY (RECLAMATION/LAND) ]\n")
                f.write(f"Maximum Elevation    : {np.max(elevation):.2f} meters\n")
                f.write(f"Average Elevation    : {np.mean(elevation):.2f} meters\n\n")
            
            f.write("Note: PIK2 is a low-lying coastal/reclamation area. Elevation \n")
            f.write("data reflects the extremely flat topography of the Jakarta Bay.\n")
            f.write("=================================================================\n")
            
        print(f"Report saved to: {report_path}")

    def generate_maps(self):
        """Generate the publication-ready 2-panel PyGMT plot."""
        print("Generating geospatial figures...")
        
        fig = pygmt.Figure()
        
        # Color palette optimized for coastal/shallow areas
        pygmt.makecpt(cmap="geo", series=[-100, 100, 1], continuous=True)

        with fig.subplot(nrows=2, ncols=1, figsize=("16c", "22c"), margins=["0.5c", "0.8c"]):
            
            # --- PANEL A: INDONESIA GENERAL ---
            with fig.set_panel(panel=0):
                fig.basemap(region=self.reg_indo, projection="M?", frame=["WSne", "xa10f5", "ya5f1"])
                
                grid_indo = pygmt.datasets.load_earth_relief(resolution="10m", region=self.reg_indo)
                fig.grdimage(grid=grid_indo, cmap=True, shading=True)

                # Plot the PIK2 Area Box (Black)
                study_x = [self.reg_study[0], self.reg_study[1], self.reg_study[1], self.reg_study[0], self.reg_study[0]]
                study_y = [self.reg_study[2], self.reg_study[2], self.reg_study[3], self.reg_study[3], self.reg_study[2]]
                fig.plot(x=study_x, y=study_y, pen="1.5p,black")
                
                # Marker for PIK2
                fig.plot(x=self.pik2_lon, y=self.pik2_lat, style="c0.3c", fill="red", pen="0.5p,black")
                fig.text(position="TC", text="(a) Indonesia Context", font="14p,Helvetica-Bold", justify="BC", offset="0c/0.3c", no_clip=True)

            # --- PANEL B: PIK2 DETAIL (No impact/control zones) ---
            with fig.set_panel(panel=1):
                # Using a larger scale for the zoomed-in area
                fig.basemap(region=self.reg_study, projection="M?", frame=["WSne", "xa0.05f0.01", "ya0.02f0.01"])
                
                fig.grdimage(grid=self.grid_study, cmap=True, shading=True)
                
                # Simple Label for PIK2
                fig.plot(x=self.pik2_lon, y=self.pik2_lat, style="a0.5c", fill="yellow", pen="0.5p,black")
                fig.text(x=self.pik2_lon, y=self.pik2_lat - 0.01, text="PIK2 Area", font="12p,Helvetica-Bold,black", justify="TC")
                
                fig.text(position="TC", text="(b) PIK2 Local Topography", font="14p,Helvetica-Bold", justify="BC", offset="0c/0.3c", no_clip=True)

        # Global Colorbar
        with pygmt.config(FONT_LABEL="13p,Helvetica-Bold", MAP_LABEL_OFFSET="10p"):
            fig.colorbar(
                cmap=True, 
                position="JRM+jMC+w16c/0.6c+o3.0c/0c", 
                frame=["x+lElevation", "y+lmeters"]
            )

        # Save
        png_path = os.path.join(self.figs_dir, "PIK2_Geospatial_Map.png")
        fig.savefig(png_path, dpi=400)
        print(f"Map saved to: {png_path}")

    def run(self):
        self.load_grids_safely()
        self.generate_bathymetry_report()
        self.generate_maps()
        print("\nPIK2 Mapping Pipeline finished!")

if __name__ == "__main__":
    generator = PIK2MapGenerator()
    generator.run()
