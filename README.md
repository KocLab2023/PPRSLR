## Privacy-Preserving Recommendation System based on Logistic Regression using Fully Homomorphic Encryption

A Go implementation of privacy-preserving recommendation system based on logistic regression using the CKKS fully homomorphic encryption scheme.
This project demonstrates how to perform training and inference directly on encrypted data, enabling secure recommendation system in privacy-sensitive environments.

### Features
CKKS-based encrypted computation using Lattigo v6.
	•	End-to-end workflow: load CSV → encode → encrypt → train → predict.  
	•	Encrypted linear algebra pipeline: rotations, linear transformations, rescaling, relinearization.  
	•	Polynomial sigmoid approximation for encrypted logistic regression.

### Project Structure
.
├─ main.go        # Entry point: args parsing, data loading, model training/testing  
├─ LR.go          # Core logistic regression logic on encrypted data  
└─ utils.go       # Helper functions (CSV loading, sigmoid approximation, lintrans utilities)

### Dependencies
	•	Go ≥ 1.18
	•	Lattigo v6
