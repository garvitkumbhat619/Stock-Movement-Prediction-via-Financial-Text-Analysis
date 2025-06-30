import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "The market is crashing badly today!"}

response = requests.post(url, json=data)
print(response.json())
