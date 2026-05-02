import unittest
from app import app

class TestFunctional(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_full_flow(self):
        # Step 1: Login
        login = self.client.post('/login', data={
            'username': 'admin',
            'password': 'admin'
        }, follow_redirects=True)

        self.assertEqual(login.status_code, 200)

        # Step 2: Access dashboard
        dashboard = self.client.get('/home')
        self.assertEqual(dashboard.status_code, 200)

        # Step 3: Crop Prediction
        prediction = self.client.post('/predict', data={
            'Nitrogen': '60',
            'Phosphorus': '30',
            'Potassium': '20',
            'Temperature': '28',
            'pH': '6.8'
        })

        self.assertEqual(prediction.status_code, 200)
        self.assertIn(b'Recommended Crop', prediction.data)

if __name__ == '__main__':
    unittest.main()
