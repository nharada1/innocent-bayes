//
// Created by Nate Harada on 3/3/16.
//

#include "NaiveBayes.h"



void NaiveBayes::fit(MatrixXd X, VectorXi y) {
    int n_classes = y.maxCoeff() + 1;
    int n_features = X.cols();

    pC.resize(n_classes);
    pXC.resize(n_features, n_classes);

    // Compute class probabilites P(C_k)
    for (int c=0; c<n_classes; c++) {
        pC(c) = getClassProb(y, c);
    }

    // Compute class-conditional probabilities P(x_i | C_k)
    for (int i=0; i<n_features; i++) {
        for (int c=0; c<n_classes; c++) {
            pXC(i, c) = getClassCondProb(X, y, i, c);
        }
    }

}

MatrixXd NaiveBayes::predict_proba(MatrixXd X) {
    // Naive bayes can be viewed as a linear classifier when expressed in log space
    // where pC is the bias
    MatrixXd rel_probs = X * pXC;
    rel_probs.rowwise() += pC.transpose();

    return rel_probs;
}

VectorXi NaiveBayes::predict(MatrixXd X) {
    MatrixXd rel_probs = predict_proba(X);
    VectorXi maxVal(rel_probs.rows());
    for (int r=0; r<rel_probs.rows(); ++r) {
        float max = -std::numeric_limits<float>::max();
        int argmax = 0;

        // Calculate the max prob for this row
        for (int c=0; c<rel_probs.cols(); ++c) {
            if (rel_probs(r, c) > max) {
                max = rel_probs(r, c);
                argmax = c;
            }
        }
        maxVal(r) = argmax;
    }
    std::cout << rel_probs << "\n";
    std::cout << maxVal << "\n";
    return maxVal;
}

float NaiveBayes::getClassProb(VectorXi v, int c) {
    int n_class = 0;

    for (int i=0; i<v.size(); ++i) {
        if (v(i) == c)
            n_class++;
    }
    return log(float(n_class) / v.size());
}

float NaiveBayes::getClassCondProb(MatrixXd X, VectorXi y, int i, int c) {
    float Nyi = 0;
    float Ny = 0;
    float alpha = 1;

    // Compute number of times feature appears in class y
    // Compute sum of features in class y
    for (int r=0; r<X.rows(); r++) {
        if (y(r) == c) {
            Nyi += X(r, i);
            for (int f=0; f<X.cols(); f++) {
                Ny += X(r, f);
            }
        }
    }
    float smoothed = (Nyi + alpha) / (Ny + alpha*y.maxCoeff());
    return log(smoothed);
}


