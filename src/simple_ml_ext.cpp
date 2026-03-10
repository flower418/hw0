#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, 
                  float *theta, size_t m, size_t n, size_t k, 
                  float lr, size_t batch) 
{
    // 遍历每一个 batch
    for (size_t i = 0; i < m; i += batch) {
        // 处理最后一个 batch 可能不足 batch 大小的情况
        size_t current_batch = (i + batch > m) ? (m - i) : batch;
        
        // 1. 计算 logits: Z = X_batch * theta (current_batch x k)
        // 我们需要一块临时空间存储当前 batch 的结果
        float *Z = new float[current_batch * k];
        
        for (size_t bi = 0; bi < current_batch; bi++) {
            size_t row_idx = i + bi; // X 中真实的行索引
            for (size_t bj = 0; bj < k; bj++) {
                float sum = 0;
                for (size_t bl = 0; bl < n; bl++) {
                    sum += X[row_idx * n + bl] * theta[bl * k + bj];
                }
                Z[bi * k + bj] = exp(sum); // 直接取 exp
            }
            
            // 归一化 (Softmax 步骤)
            float row_sum = 0;
            for (size_t bj = 0; bj < k; bj++) row_sum += Z[bi * k + bj];
            for (size_t bj = 0; bj < k; bj++) Z[bi * k + bj] /= row_sum;
        }

        // 2. 计算梯度并更新: grad = X_batch.T * (Z - I_y) / batch
        // Z 现在是 softmax 后的结果，我们需要 Z - I_y
        for (size_t bi = 0; bi < current_batch; bi++) {
            size_t row_idx = i + bi;
            Z[bi * k + y[row_idx]] -= 1.0f; // 相当于 Z - I_y
        }

        // 计算梯度并直接更新到 theta 上：theta -= lr * (X_batch.T @ Z) / batch
        for (size_t j = 0; j < n; j++) {
            for (size_t l = 0; l < k; l++) {
                float grad_element = 0;
                for (size_t bi = 0; bi < current_batch; bi++) {
                    // X_batch.T 的第 (j, bi) 是 X 的第 (row_idx, j)
                    grad_element += X[(i + bi) * n + j] * Z[bi * k + l];
                }
                // 原地修改 theta
                theta[j * k + l] -= lr * grad_element / (float)current_batch;
            }
        }

        delete[] Z; // 释放临时空间
    }
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
