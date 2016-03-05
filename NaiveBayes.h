//
// Created by Nate Harada on 3/3/16.
//

#ifndef INNOCENT_BAYES_NAIVEBAYES_H
#define INNOCENT_BAYES_NAIVEBAYES_H

#include <set>
#include <iostream>
#include <limits.h>
#include <math.h>

#include "Eigen/Core"

using namespace Eigen;

class NaiveBayes {
public:
    NaiveBayes() {};

    static float getClassProb(VectorXi v, int c);
    static float getClassCondProb(MatrixXd X, VectorXi y, int i, int c);

    void fit(MatrixXd X, VectorXi y);
    MatrixXd predict_proba(MatrixXd X);
    VectorXi predict(MatrixXd X);

private:
    // Prob of classes in the training set
    VectorXd pC;
    // Accessing (i, k) gives P(x_i|C_k), where i is the feature, k is the class
    MatrixXd pXC;
};


#endif //INNOCENT_BAYES_NAIVEBAYES_H
