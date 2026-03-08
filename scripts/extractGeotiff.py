#!/usr/bin/env python3
"""
================================================================
  Sentinel-2 LULC  →  NetCDF4  |  PIK2, Indonesia
  Author : Sandy H. S. Herho <sandy.herho@email.ucr.edu>
  Date   : 2026-03-08
  Region : PIK2 (lon 106.63–106.77, lat -6.08–-5.98)
  License: MIT

  Esri Sentinel-2 10-m Annual LULC class codes
  ─────────────────────────────────────────────
    1  = Water              7  = Built Area
    2  = Trees              8  = Bare Ground
    4  = Flooded Vegetation 9  = Snow / Ice
    5  = Crops             10  = Clouds
                           11  = Rangeland

  Input  : ../geotiff/48M_YYYYMMDD-YYYYMMDD.tif  (one file / year)
  Output : ../netcdf/pik2LULC.nc
================================================================
"""

import os, re
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as tfrom_bounds
import netCDF4 as nc
from datetime import datetime, timezone
from pyproj import Transformer

# ── Settings ──────────────────────────────────────────────────
GEOTIFF_DIR = "../geotiff"
OUTFILE     = "../netcdf/pik2LULC.nc"
ROW_CHUNK   = 256          # rows per chunk — keeps peak RAM < ~200 MB

# PIK2 bounding box (WGS-84)
LON_MIN, LON_MAX = 106.63, 106.77
LAT_MIN, LAT_MAX =  -6.08,  -5.98

CLASS_VALUES = np.array([1, 2, 4, 5, 7, 8, 9, 10, 11], dtype=np.int8)
CLASS_NAMES  = ["Water", "Trees", "Flooded_Vegetation", "Crops",
                "Built_Area", "Bare_Ground", "Snow_Ice", "Clouds", "Rangeland"]
CLASS_COLORS = ["#419BDF", "#397D49", "#7A87C6", "#E49635", "#C4281B",
                "#A59B8F", "#FFFFFF", "#B3B3B3", "#E3E2C3"]
FILL_VAL = np.int8(-128)
EPOCH    = datetime(1970, 1, 1)

# ── Helpers ───────────────────────────────────────────────────

def parse_year(fname):
    m = re.search(r"_(\d{4})\d{4}-", os.path.basename(fname))
    if not m:
        raise ValueError(f"Cannot parse year from '{fname}'")
    return int(m.group(1))


def collect_tifs(d):
    pat = re.compile(r"48M_\d{8}-\d{8}\.tif$", re.IGNORECASE)
    pairs = sorted(
        [(parse_year(f), os.path.join(d, f))
         for f in os.listdir(d) if pat.match(f)])
    if not pairs:
        raise FileNotFoundError(f"No matching 48M_*.tif in '{d}'")
    return pairs


def build_target_grid(src):
    """
    Build a regular WGS-84 target grid that matches the source ~10 m resolution.
    Returns (lon_1d, lat_1d)  — lat descending (north → south).
    """
    # estimate native pixel size in degrees from source transform
    native_m = abs(src.transform.a)          # metres per pixel (approx)
    deg_per_px = native_m / 111_320.0        # rough conversion

    lon_1d = np.arange(LON_MIN + deg_per_px/2,
                       LON_MAX,
                       deg_per_px, dtype=np.float64)
    lat_1d = np.arange(LAT_MAX - deg_per_px/2,
                       LAT_MIN,
                       -deg_per_px, dtype=np.float64)
    return lon_1d, lat_1d


def read_strip(fpath, src_crs, needs_warp,
               lon_1d, lat_1d, r0, r1):
    """
    Read rows r0:r1 of the target grid from one GeoTIFF.
    Reprojects to WGS-84 if needed.
    """
    n_lon  = lon_1d.size
    n_row  = r1 - r0
    lat_strip = lat_1d[r0:r1]           # descending subset

    lat_top = lat_strip[0]  + abs(lat_1d[1] - lat_1d[0]) / 2
    lat_bot = lat_strip[-1] - abs(lat_1d[1] - lat_1d[0]) / 2
    lon_lft = lon_1d[0]  - abs(lon_1d[1] - lon_1d[0]) / 2
    lon_rgt = lon_1d[-1] + abs(lon_1d[1] - lon_1d[0]) / 2

    dst_transform = tfrom_bounds(lon_lft, lat_bot, lon_rgt, lat_top,
                                 n_lon, n_row)
    dst = np.full((n_row, n_lon), FILL_VAL, dtype=np.int8)

    with rasterio.open(fpath) as src:
        if needs_warp:
            # transform target bbox corners → source CRS to find window
            tr = Transformer.from_crs("EPSG:4326", src_crs.to_epsg(),
                                      always_xy=True)
            corners_x = [lon_lft, lon_lft, lon_rgt, lon_rgt]
            corners_y = [lat_bot, lat_top, lat_bot, lat_top]
            sx, sy = tr.transform(corners_x, corners_y)
            src_win = from_bounds(min(sx), min(sy), max(sx), max(sy),
                                  src.transform)
            src_win = src_win.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height))
            src_data = src.read(1, window=src_win).astype(np.int8)
            src_transform = src.window_transform(src_win)
        else:
            src_win = from_bounds(lon_lft, lat_bot, lon_rgt, lat_top,
                                  src.transform)
            src_win = src_win.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height))
            src_data = src.read(1, window=src_win).astype(np.int8)
            src_transform = src.window_transform(src_win)

    reproject(
        source=src_data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=CRS.from_epsg(4326),
        resampling=Resampling.nearest,
    )
    return dst


# ── Main ──────────────────────────────────────────────────────

def main():
    outdir = os.path.dirname(OUTFILE)
    os.makedirs(outdir, exist_ok=True)

    # ensure the output directory is writable
    if not os.access(outdir, os.W_OK):
        raise PermissionError(f"Output directory not writable: '{outdir}'\n"
                              f"  Try: chmod u+w {outdir}")

    # remove stale / partial file from a previous run
    if os.path.exists(OUTFILE):
        try:
            os.remove(OUTFILE)
            print(f"Removed stale file: {OUTFILE}")
        except OSError as e:
            raise PermissionError(
                f"Cannot overwrite '{OUTFILE}': {e}\n"
                f"  Try: rm -f {OUTFILE}") from e
    tifs = collect_tifs(GEOTIFF_DIR)
    print(f"Found {len(tifs)} file(s):")
    for yr, fp in tifs:
        print(f"  {yr}  {os.path.basename(fp)}")

    with rasterio.open(tifs[0][1]) as src0:
        src_crs    = src0.crs
        needs_warp = (src_crs != CRS.from_epsg(4326))
        lon_arr, lat_arr = build_target_grid(src0)

    n_lat, n_lon = lat_arr.size, lon_arr.size
    n_time       = len(tifs)
    years        = [yr for yr, _ in tifs]
    time_vals    = np.array([(datetime(yr,1,1)-EPOCH).days for yr in years],
                            dtype=np.int32)

    print(f"\nGrid : {n_lat} lat × {n_lon} lon")
    print(f"BBox : lon [{lon_arr[0]:.4f}, {lon_arr[-1]:.4f}]  "
          f"lat [{lat_arr[-1]:.4f}, {lat_arr[0]:.4f}]")
    print(f"Writing → {OUTFILE}\n")

    # ── create NetCDF skeleton ────────────────────────────────
    with nc.Dataset(OUTFILE, "w", format="NETCDF4") as ds:

        # global attributes
        ds.title       = ("Sentinel-2 Annual LULC — "
                          "PIK2, Indonesia")
        ds.source      = ("Esri Sentinel-2 10-m Annual Land Use / "
                          "Land Cover, tile 48M")
        ds.institution = ("Department of Earth and Planetary Sciences, "
                          "University of California, Riverside")
        ds.author      = "Sandy H. S. Herho"
        ds.contact     = "sandy.herho@email.ucr.edu"
        ds.Conventions = "CF-1.8"
        ds.license     = "MIT"
        ds.geospatial_lat_min = float(LAT_MIN)
        ds.geospatial_lat_max = float(LAT_MAX)
        ds.geospatial_lon_min = float(LON_MIN)
        ds.geospatial_lon_max = float(LON_MAX)
        ds.date_created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ds.history      = (f"{datetime.now(timezone.utc):%Y-%m-%d} "
                           "Created by geotiff_to_lulc_netcdf.py")

        # dimensions
        ds.createDimension("time",      n_time)
        ds.createDimension("lat",       n_lat)
        ds.createDimension("lon",       n_lon)
        ds.createDimension("n_classes", len(CLASS_VALUES))

        # time
        vt           = ds.createVariable("time", "i4", ("time",))
        vt.units     = "days since 1970-01-01"
        vt.calendar  = "gregorian"
        vt.long_name = "time"
        vt.axis      = "T"
        vt[:]        = time_vals

        vy           = ds.createVariable("year", "i2", ("time",))
        vy.long_name = "calendar year of annual composite"
        vy.units     = "year"
        vy[:]        = np.array(years, dtype=np.int16)

        # lat / lon  — NO valid_min/valid_max to prevent coordinate masking
        vlat               = ds.createVariable("lat", "f8", ("lat",))
        vlat.units         = "degrees_north"
        vlat.standard_name = "latitude"
        vlat.long_name     = "latitude"
        vlat.axis          = "Y"
        vlat[:]            = lat_arr

        vlon               = ds.createVariable("lon", "f8", ("lon",))
        vlon.units         = "degrees_east"
        vlon.standard_name = "longitude"
        vlon.long_name     = "longitude"
        vlon.axis          = "X"
        vlon[:]            = lon_arr

        # LULC — chunked + compressed
        vlulc = ds.createVariable(
            "lulc", "i1", ("time", "lat", "lon"),
            chunksizes=(1, 256, 256),
            zlib=True, complevel=6,
            fill_value=FILL_VAL)
        vlulc.long_name     = ("Annual land use / land cover class, "
                               "PIK2")
        vlulc.standard_name = "land_cover"
        vlulc.units         = "1"
        vlulc.grid_mapping  = "crs"
        vlulc.flag_values   = CLASS_VALUES
        vlulc.flag_meanings = " ".join(CLASS_NAMES)
        vlulc.comment       = ("Esri LULC codes: 1=Water 2=Trees "
                               "4=Flooded_Vegetation 5=Crops 7=Built_Area "
                               "8=Bare_Ground 9=Snow_Ice 10=Clouds "
                               "11=Rangeland.  -128=no-data.")

        # class lookup table
        vcv           = ds.createVariable("class_value", "i1", ("n_classes",))
        vcv.long_name = "LULC class integer code"
        vcv[:]        = CLASS_VALUES

        vlen_str = ds.createVLType(str, "vlen_str")

        vcn           = ds.createVariable("class_name", vlen_str, ("n_classes",))
        vcn.long_name = "LULC class description"
        vcn[:]        = np.array(CLASS_NAMES, dtype=object)

        vcc           = ds.createVariable("class_color", vlen_str, ("n_classes",))
        vcc.long_name = "Suggested hex colour for visualisation"
        vcc[:]        = np.array(CLASS_COLORS, dtype=object)

        # CRS (WGS-84)
        vcrs = ds.createVariable("crs", "i4")
        vcrs.grid_mapping_name            = "latitude_longitude"
        vcrs.longitude_of_prime_meridian  = 0.0
        vcrs.semi_major_axis              = 6378137.0
        vcrs.inverse_flattening           = 298.257223563
        vcrs.crs_wkt = ('GEOGCS["WGS 84",DATUM["WGS_1984",'
                        'SPHEROID["WGS 84",6378137,298.257223563]],'
                        'PRIMEM["Greenwich",0],'
                        'UNIT["degree",0.0174532925199433]]')
        vcrs.epsg_code = "EPSG:4326"

        # ── stream data row-chunk by row-chunk ────────────────
        total_chunks = (n_lat + ROW_CHUNK - 1) // ROW_CHUNK
        for ci, r0 in enumerate(range(0, n_lat, ROW_CHUNK), 1):
            r1  = min(r0 + ROW_CHUNK, n_lat)
            buf = np.full((n_time, r1-r0, n_lon), FILL_VAL, dtype=np.int8)

            for t, (yr, fp) in enumerate(tifs):
                strip = read_strip(fp, src_crs, needs_warp,
                                   lon_arr, lat_arr, r0, r1)
                r = min(strip.shape[0], r1-r0)
                c = min(strip.shape[1], n_lon)
                buf[t, :r, :c] = strip[:r, :c]

            ds["lulc"][:, r0:r1, :] = buf
            print(f"  [{ci:3d}/{total_chunks}  {ci/total_chunks*100:5.1f}%]  "
                  f"rows {r0:5d}–{r1:5d}", flush=True)

    # ── verification ──────────────────────────────────────────
    print(f"\nVerification:")
    with nc.Dataset(OUTFILE) as ds:
        lulc = ds["lulc"][:]
        lon  = ds["lon"][:].data          # .data avoids MaskedArray formatting
        lat  = ds["lat"][:].data
        print(f"  dimensions : { {k: len(v) for k,v in ds.dimensions.items()} }")
        print(f"  variables  : {list(ds.variables.keys())}")
        print(f"  years      : {list(ds['year'][:])}")
        print(f"  lulc shape : {lulc.shape}  dtype={lulc.dtype}")
        print(f"  lon range  : [{lon.min():.4f}, {lon.max():.4f}]")
        print(f"  lat range  : [{lat.min():.4f}, {lat.max():.4f}]")
        unique = np.unique(lulc.data[lulc.data != int(FILL_VAL)])
        names  = {v: n for v, n in zip(CLASS_VALUES, CLASS_NAMES)}
        print(f"  LULC codes : { {int(v): names.get(int(v), '?') for v in unique} }")
        print(f"  file size  : {os.path.getsize(OUTFILE)/1024**2:.2f} MB")

    print("\nDone.")


if __name__ == "__main__":
    main()
