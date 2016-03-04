//
// Created by Nate Harada on 3/3/16.
//

#include "NaiveBayes.h"

#include <iostream>


void NaiveBayes::fit(MatrixXf X, VectorXi y) {
    int n_classes = y.maxCoeff() + 1;
    pC.resize(n_classes);
    for (int c=0; c<n_classes; c++) {
        pC(c) = getClassProb(y, c);
    }
}

MatrixXf NaiveBayes::predict(MatrixXf X) {
    for (int i=0; i<pC.size(); ++i)
        std::cout << pC(i) << " ";

    MatrixXf predictions;
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
