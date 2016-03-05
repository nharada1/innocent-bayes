//
// Created by Nate Harada on 3/3/16.
//

#include "NaiveBayes.h"

#include <iostream>


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

MatrixXd NaiveBayes::predict(MatrixXd X) {
    for (int i=0; i<pC.size(); ++i)
        std::cout << pC(i) << " ";

    MatrixXd predictions;
    return predictions;
}

float NaiveBayes::getClassProb(VectorXi v, int c) {
    int n_class = 0;

    for (int i=0; i<v.size(); ++i) {
        if (v(i) == c)
            n_class++;
    }
    return float(n_class) / v.size();
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
    return smoothed;
}
