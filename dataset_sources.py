# dataset_sources.py
"""
Integration with various cyberbullying datasets
"""

DATASET_SOURCES = {
    "Kaggle Cyberbullying Tweets": {
        "url": "https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification",
        "columns": {
            "text": "tweet_text",
            "label": "cyberbullying_type"
        },
        "label_mapping": {
            "not_cyberbullying": 0,
            "gender": 1,
            "religion": 1,
            "other_cyberbullying": 1,
            "age": 1,
            "ethnicity": 1
        }
    },

    "Hate Speech Dataset": {
        "url": "https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset",
        "columns": {
            "text": "tweet",
            "label": "class"
        },
        "label_mapping": {
            0: 0,  # hate speech
            1: 1,  # offensive language
            2: 0  # neither
        }
    },

    "Toxic Comments": {
        "url": "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge",
        "columns": {
            "text": "comment_text",
            "label": "toxic"
        },
        "label_mapping": {
            0: 0,
            1: 1
        }
    }
}