# DM Movie Recommendation Project

This project, A Synergistic Multi-Method Movie Recommendation System, integrates multiple techniques to build a robust, accurate, and personalized movie recommendation system. By combining Matrix Factorization (MF), Locality-Sensitive Hashing (LSH), and the Apriori algorithm, the system addresses limitations of traditional methods, such as lack of diversity, sparse data, and insufficient contextual information.

## contributors

- Huisu Lee
- Junyoung Hur
- Seongmin Kim

## Project Overview

With the rapid growth of online movie content, this project aims to create a hybrid recommendation system that blends collaborative filtering and content-based approaches. The process consists of three main steps:
1.	Matrix Factorization (MF): Collaborative filtering for rating prediction.
2.	Locality-Sensitive Hashing (LSH): Clustering movies based on metadata similarity.
3.	Apriori Algorithm: Extracting frequent patterns in movie attributes to refine recommendations.

These techniques are combined to deliver more accurate, diverse, and contextually relevant movie suggestions.

## Files

### 1. data_visualiza.ipynb

This notebook performs exploratory data analysis (EDA) and visualizations on the dataset to justify the use of MF, LSH, and Apriori.
- Key Features:
    - Visualizes rating distributions (overall, per user, per movie) to highlight sparsity and bias.
    - Analyzes metadata distributions, including genres, budgets, vote counts, and runtime.
    - Filters and preprocesses data for downstream tasks (e.g., normalizing ratings and cleaning metadata).
- Output:
    - Figures and insights that guide the feature selection and preprocessing process.

### 2. apriori.ipynb

This notebook applies the Apriori Algorithm to identify frequent patterns among movies based on their attributes.
- Key Features:
    - Uses runtime, genre, and production company to detect frequent itemsets.
    - Filters less frequent attributes to retain the top 90%.
    - Generates association rules to uncover patterns in movie co-occurrences.
- Output:
    - Association rules used to refine the recommendation process by emphasizing shared attributes among highly rated movies.

### 3. locally_sensitive_hashing.ipynb

This notebook implements Locality-Sensitive Hashing (LSH) to group movies with similar metadata efficiently.
- Key Features:
    - Processes movie attributes such as genres, production companies, and textual overviews.
    - TF-IDF vectorization for text-based similarity and one-hot encoding for categorical features.
    - Applies LSH to cluster movies with high similarity, enabling faster retrieval of related movies.
- Output:
    - LSH similarity sets used to filter MF recommendations based on user-rated movies $(\mathrm{rating} \ge 4)$.

### 4. matrix_factorization.ipynb

This notebook uses Matrix Factorization (MF) to predict user ratings and generate initial recommendations.
- Key Features:
    - Preprocesses rating data by normalizing ratings and filtering users/movies with low interaction.
    - Builds an MF model using latent factorization with PyTorch.
    - Configures model parameters (latent dimension: 60, loss: MSE, optimizer: Adam).
    - Evaluates performance using RMSE and accuracy thresholds.
- Output:
    - Top-N recommended movies for each user, which are later refined using LSH and Apriori filters.

## Workflow

1.	Initial Recommendation (MF):
    - Use MF to predict ratings and generate a ranked list of top-N movies for each user.
2.	Metadata-Based Filtering (LSH):
    - Identify movies similar to user-rated (≥4) movies based on metadata features.
3.	Pattern-Based Filtering (Apriori):
    - Discover frequent attribute patterns (e.g., runtime, genres) from user-rated movies and apply these associations.
4.	Final Recommendation:
    - Filter the MF-generated list using both LSH and Apriori outputs, retaining only the most contextually relevant and diverse movies.

## Conclusion

This project successfully combines Matrix Factorization, Locality-Sensitive Hashing, and Apriori into a hybrid recommendation system. By leveraging collaborative filtering, metadata similarity, and frequent item patterns, the system delivers accurate, diverse, and contextually relevant recommendations. This multi-method approach addresses key challenges like data sparsity and lack of diversity, offering a robust framework for real-world recommendation scenarios.

## Future Work
- Incorporating real-time user feedback to improve recommendations dynamically.
- Enhancing feature embeddings using deep learning techniques for better similarity modeling.