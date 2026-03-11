#!/usr/bin/env python
"""
PIK2 Coastal and Regional Geospatial Context Map
Generates a high-resolution 2-panel map (Indonesia & PIK2) using PyGMT.
Extracts and reports bathymetry and elevation statistics.

Author : Sandy H. S. Herho
Date   : 2026/02/22
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
        self.reg_indo = [90.0, 145.0, -13.0, 8.0]               # Panel A: Indonesia General (expanded)
        self.reg_study = [106.63, 106.77, -6.08, -5.98]          # Panel B: PIK2 Region

        # PIK2 centroid for marker on Panel A
        self.pik2_lon = (106.63 + 106.77) / 2   # 106.70
        self.pik2_lat = (-6.08 + -5.98) / 2     # -6.03

        # 3. Data Grids Placeholder
        self.grid_study = None

    def _setup_directories(self):
        """Create necessary output directories if they don't exist."""
        for directory in [self.figs_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory verified: {directory}")

    def load_grids_safely(self):
        """Loads grids naturally with error catching for corrupted network cache."""
        try:
            print("Downloading/Loading 1 arc-second high-res grid for the PIK2 study area...")
            self.grid_study = pygmt.datasets.load_earth_relief(
                resolution="01s",   # 1 arc-second (~30m) — sharpest available
                region=self.reg_study
            )
        except ValueError:
            print("\n[CRITICAL ERROR] Corrupted GMT Cache Detected.")
            print("A previous network timeout left a corrupted, 0-byte file on your machine.")
            print("Please open your terminal and run the following command to fix this:\n")
            print("    rm -rf ~/.gmt/server/earth/earth_relief\n")
            print("Then run this script again.")
            sys.exit(1)

    def generate_bathymetry_report(self):
        """Extract high-resolution bathymetry data and generate a statistical report."""
        values = self.grid_study.values
        ocean_mask = values < 0
        land_mask = values >= 0

        bathymetry = values[ocean_mask]
        elevation = values[land_mask]

        report_path = os.path.join(self.reports_dir, "Bathymetry_Topography_Report.txt")

        with open(report_path, "w") as f:
            f.write("=================================================================\n")
            f.write(" GEOMORPHOLOGICAL REPORT: STUDY AREA (PIK2 Region)\n")
            f.write(" Coordinate Bounding Box : Lon [106.63 to 106.77], Lat [-6.08 to -5.98]\n")
            f.write(" Dataset Resolution      : 1 arc-second (~30 meters)\n")
            f.write(" Author: Sandy H. S. Herho | Date: 2026/02/22\n")
            f.write("=================================================================\n\n")

            f.write("[ MARINE BATHYMETRY (OCEAN) ]\n")
            if len(bathymetry) > 0:
                f.write(f"Total Ocean Pixels  : {len(bathymetry):,}\n")
                f.write(f"Maximum Depth       : {np.min(bathymetry):.2f} meters\n")
                f.write(f"Average Depth (Mean): {np.mean(bathymetry):.2f} meters\n")
                f.write(f"Median Depth        : {np.median(bathymetry):.2f} meters\n\n")
            else:
                f.write("No ocean pixels detected in this region.\n\n")

            f.write("[ TERRESTRIAL TOPOGRAPHY (LAND) ]\n")
            if len(elevation) > 0:
                f.write(f"Total Land Pixels   : {len(elevation):,}\n")
                f.write(f"Maximum Elevation   : {np.max(elevation):.2f} meters\n")
                f.write(f"Average Elevation   : {np.mean(elevation):.2f} meters\n\n")
            else:
                f.write("No land pixels detected in this region.\n\n")

            f.write("=================================================================\n")

        print(f"Bathymetry Report successfully saved to: {report_path}")

    def generate_maps(self):
        """Generate the publication-ready 2-panel PyGMT plot."""
        print("Generating geospatial figures...")

        fig = pygmt.Figure()

        # Global color palette for Earth relief
        pygmt.makecpt(cmap="geo", series=[-6000, 3000, 100], continuous=True)

        # 2x1 subplot structure
        with fig.subplot(nrows=2, ncols=1, figsize=("16c", "22c"), margins=["0.5c", "0.8c"]):

            # ---------------------------------------------------------
            # PANEL A: INDONESIA GENERAL
            # ---------------------------------------------------------
            with fig.set_panel(panel=0):
                fig.basemap(region=self.reg_indo, projection="M?", frame=["WSne", "xa10f5", "ya5f1"])

                try:
                    grid_indo = pygmt.datasets.load_earth_relief(resolution="10m", region=self.reg_indo)
                except ValueError:
                    print("\n[CRITICAL ERROR] Corrupted GMT Cache Detected during Panel A generation.")
                    print("Please run: rm -rf ~/.gmt/server/earth/earth_relief")
                    sys.exit(1)

                fig.grdimage(grid=grid_indo, cmap=True, shading=True)

                # Plot the PIK2 Study Area Box (Black dashed) for macro context
                study_x = [self.reg_study[0], self.reg_study[1], self.reg_study[1], self.reg_study[0], self.reg_study[0]]
                study_y = [self.reg_study[2], self.reg_study[2], self.reg_study[3], self.reg_study[3], self.reg_study[2]]
                fig.plot(x=study_x, y=study_y, pen="1.5p,black,-")

                # Plot PIK2 centroid as a red dot with label
                fig.plot(x=self.pik2_lon, y=self.pik2_lat, style="c0.35c", fill="red", pen="0.5p,black")
                fig.text(x=self.pik2_lon, y=self.pik2_lat, text="PIK2", font="9p,Helvetica-Bold,black",
                         justify="LM", offset="0.2c/0c", fill="white@30")

                # Bold centered (a) label above the panel
                fig.text(position="TC", text="(a)", font="14p,Helvetica-Bold,black", justify="BC", offset="0c/0.3c", no_clip=True)

            # ---------------------------------------------------------
            # PANEL B: PIK2 REGION (1 arc-second, ~30m resolution)
            # ---------------------------------------------------------
            with fig.set_panel(panel=1):
                fig.basemap(region=self.reg_study, projection="M?", frame=["WSne", "xa0.05f0.025", "ya0.05f0.025"])

                fig.grdimage(grid=self.grid_study, cmap=True, shading=True)

                # Bold centered (b) label above the panel
                fig.text(position="TC", text="(b)", font="14p,Helvetica-Bold,black", justify="BC", offset="0c/0.3c", no_clip=True)

        # ---------------------------------------------------------
        # GLOBAL COLORBAR
        # ---------------------------------------------------------
        with pygmt.config(
            FONT_LABEL="13p,Helvetica-Bold",
            FONT_ANNOT_PRIMARY="10p,Helvetica",
            MAP_LABEL_OFFSET="10p"
        ):
            fig.colorbar(
                cmap=True,
                position="JRM+jMC+w16c/0.6c+o3.0c/0c",
                frame=["x+lElevation / Bathymetry", "y+lmeters"]
            )

        # ---------------------------------------------------------
        # SAVE FIGURES
        # ---------------------------------------------------------
        pdf_path = os.path.join(self.figs_dir, "Study_Area_Map.pdf")
        png_path = os.path.join(self.figs_dir, "Study_Area_Map.png")

        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=400)

        print(f"High-resolution maps saved successfully to: {self.figs_dir}")

    def run(self):
        """Execute the mapping and reporting pipeline."""
        print("Starting PyGMT Geospatial Mapping Pipeline...")
        self.load_grids_safely()
        self.generate_bathymetry_report()
        self.generate_maps()
        print("\nPipeline execution finished successfully!")

if __name__ == "__main__":
    generator = PIK2MapGenerator()
    generator.run()
