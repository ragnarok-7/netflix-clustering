# Netflix Movies & TV Shows Clustering

## 📌 Project Overview
This project performs **unsupervised clustering** of Netflix movies and TV shows using metadata (genre, release year, duration, description, etc.).  
The goal is to discover meaningful groups of titles (e.g., genre-based clusters, time-based clusters) and visualize them.

The workflow:
1. Data cleaning & preprocessing  
2. Feature extraction (TF-IDF on descriptions, genre encoding, numeric features)  
3. Dimensionality reduction (Truncated SVD)  
4. Clustering with KMeans  
5. Evaluation (silhouette score, Davies-Bouldin index)  
6. Visualization and cluster interpretation  

---

## 📂 Repository Structure
