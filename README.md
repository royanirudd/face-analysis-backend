# face-analysis-backend

## Run it locally and test using curl

```bash
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

curl -v -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -F "file=@image.jpg"
```

## Test using the local.html

Once the server is running on port 8000, open local.html and run app.  
Ensure server is running on 8000, alternatively you can change this port in local.html
```js
const BASE_URL = 'http://localhost:8000/analyze';
```
