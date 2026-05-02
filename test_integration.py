import unittest
from app import app

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_login_and_dashboard(self):
        response = self.client.post('/login', data={
            'username': 'admin',
            'password': 'admin'
        }, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'dashboard', response.data.lower())

    def test_weather_api(self):
        response = self.client.get('/weather')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'weather', response.data)

    def test_soil_health_live(self):
        response = self.client.get('/soil-health-live')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'success', response.data)

    def test_prediction_route(self):
        response = self.client.post('/predict', data={
            'Nitrogen': '50',
            'Phosphorus': '40',
            'Potassium': '30',
            'Temperature': '25',
            'pH': '6.5'
        })
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
