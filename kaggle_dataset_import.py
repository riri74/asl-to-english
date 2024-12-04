# gets the kaggle dataset
# to be ran once

import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('ayuraj/asl-dataset', path='.', unzip=True)