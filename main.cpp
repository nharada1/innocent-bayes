#include <iostream>

#include "NaiveBayes.h"

#include "Eigen/Core"

using namespace std;

int main() {
    MatrixXf X(10, 2);
    VectorXi y(10);

    X << 1.0, 2.2,
         1.2, 2.3,
         2.1, 2.4,
         1.4, 3.0,
         1.2, 6.5,
         2.4, 2.1,
         2.0, 3.5,
         2.2, 2.2,
         0.9, 1.9,
         0.4, 2.0;

    y << 0, 0, 1, 0, 0, 1, 1, 1, 0, 0;

    NaiveBayes model;
    model.fit(X, y);
    model.predict(X);
}