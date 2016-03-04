//
// Created by Nate Harada on 3/3/16.
//

#ifndef INNOCENT_BAYES_NAIVEBAYES_H
#define INNOCENT_BAYES_NAIVEBAYES_H

#include <set>

#include "Eigen/Core"

using namespace Eigen;

class NaiveBayes {
public:
    NaiveBayes() {};

    static float getClassProb(VectorXi v, int c);

    void fit(MatrixXf X, VectorXi y);
    MatrixXf predict(MatrixXf X);

private:
    VectorXd pC;
};


#endif //INNOCENT_BAYES_NAIVEBAYES_H
