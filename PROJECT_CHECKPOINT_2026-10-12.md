# AgroVision Project Checkpoint

Last saved: 2026-10-11 19:47:20

## Current progress
- Updated app.py with a seven-day historical rainfall estimator.
- Location selectors: State -> District -> Taluka.
- Plant Disease Detection section and existing theme preserved.
- MobileNetV2 classification and YOLO localization retained.
- Rainfall values are historical-data-based estimates, not guaranteed forecasts.
- Tomorrow: restart Streamlit, generate a new ngrok URL, test, and make small changes only.

## Important files
- [FOUND] app/app.py
- [FOUND] app/app_before_rainfall_update.py
- [FOUND] models/plant_disease_mobilenetv2.keras
- [FOUND] models/plant_disease_class_names.json
- [FOUND] outputs/yolo_training/disease_localization_v1/weights/best.pt
- [FOUND] rainfall/trained_model/rainfall_climatology_progress.csv
- [FOUND] rainfall/trained_model/ap_ts_rainfall_locations.json
