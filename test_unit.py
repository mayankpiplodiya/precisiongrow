import unittest
from app import get_weather, get_live_data, search_web

class TestUnitFunctions(unittest.TestCase):

    def test_weather_function(self):
        data = get_weather("Indore")
        self.assertIsNotNone(data)
        self.assertIn("temp", data)
        self.assertIn("humidity", data)

    def test_search_web(self):
        result = search_web("best crops for black soil")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_live_data(self):
        data, rec = get_live_data()
        # Data can be None if ThingSpeak fails → handle safely
        if data:
            self.assertIn("temperature", data)
            self.assertIn("soil_moisture", data)

if __name__ == '__main__':
    unittest.main()
