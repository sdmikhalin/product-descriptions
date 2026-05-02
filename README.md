# Automatic Product Description Generation from Images

Fine-tuning the BLIP model for generating descriptions of fashion products
from images. Built using [HSE DLA / EPFL CS-433 PyTorch project template](https://github.com/Blinorot/pytorch_project_template).

## Results

| Metric    | Score  |
|-----------|--------|
| BLEU-1    | 0.4988 |
| BLEU-4    | 0.1917 |
| ROUGE-L   | 0.5856 |
| BERTScore | 0.9073 |


## Data

Download Fashion Product Images dataset:

\`\`\`bash
kaggle datasets download -d paramaggarwal/fashion-product-images-small --unzip
\`\`\`
