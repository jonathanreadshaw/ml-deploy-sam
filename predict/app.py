import json
import joblib
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load model from file on lambda boot
model_pipeline = joblib.load('model_pipeline.joblib')


def lambda_handler(event, context):
    # Load body and log content
    body = json.loads(event['body'])
    logger.info('Received request body: {}'.format(event['body']))

    # Check features are provided
    if 'features' not in body.keys():
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "message": "Invalid request. Missing parameters in body",
                }
            ),
        }

    features = np.array(body['features']).reshape(1, -1)

    # Check feature dimensions
    if features.shape[-1] != 4:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "message": "Invalid request. Received {} parameters, expected 4.".format(features.shape[1]),
                }
            ),
        }

    # Calculate prediction
    try:
        prediction = model_pipeline.predict(features)
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "prediction": str(prediction[0]),
                }
            ),
        }

    except Exception as e:
        logger.error('Unhandled error: {}'.format(e))
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Unhandled error.",
                }
            ),
        }


