# Marketing ROI Optimization

## Structure

- notebooks/ : EDA + modeling and implementation starter
- src/ : training pipeline and utilities
- api/ : FastAPI inference service
- frontend/ : React dashboard
- artifacts/ : trained model and metrics
- reports/ : report outline and assets

## Quickstart

1. Install Python deps:
   pip install -r requirements.txt
2. Train and save the best model:
   python -m src.train
3. Run the API:
   uvicorn api.main:app --reload
4. Run the dashboard:
   cd frontend
   npm install
   npm run dev

Set the API base URL for the front end with:
VITE_API_BASE_URL=http://localhost:8000
