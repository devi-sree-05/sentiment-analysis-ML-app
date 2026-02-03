1. Clone the repo: 
git clone https://github.com/devi-sree-05/sentiment-analysis-ML-app.git
cd sentiment-analysis-ML-app

2. Create and activate virtual environment
   (WINDOWS)
   python -m venv venv
   venv\Scripts\activate

   (LINUX)
   python3 -m venv venv
   source venv/bin/activate

3. Install Dependencies
   pip install -r requirements.txt

4. Run the FastAPI Backend
   uvicorn app.main:app --reload

5. Run the frontend
   frontend/index.html

   GO LIVE !!!
   Enter text, click Analyze, and view sentiment + confidence.

6. Dataset used:
   
   Source: Kaggle
🔗 https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset


