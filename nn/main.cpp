#define _SCL_SECURE_NO_WARNINGS
#include "tiny_dnn/tiny_dnn.h"
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <math.h> 
#include <fstream>
#include <iomanip>
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
inline void write_vector(const vec_t & data, std::ofstream &ofs) {
	for (auto &it : data) {
		ofs << std::left << std::setw(10) << it << " ";
	}
	ofs << std::endl;
}
void write_tensor(const char * file_name, const tensor_t & data) {
	std::ofstream ofs(file_name, std::ofstream::out);
	for (auto &outor : data) {
		write_vector(outor, ofs);
	}
	ofs.close();
}

void get_error_stat(tensor_t & y_pred, tensor_t & y) {
	auto m = y_pred[0].size();
	auto n = y_pred.size();
	vec_t feat_diff_avg(m);
	vec_t feat_avg(m);
	vec_t error_per_sample(n);
	vec_t feat_mx(m, FLT_MIN);
	vec_t feat_mn(m, FLT_MAX);
	double sample_mx = 0, sample_mn = DBL_MAX;
	for (size_t i = 0; i < n; ++i) {
		double sample_error = 0;
		for (size_t j = 0; j < m; ++j) {
			auto diff = std::abs(y_pred[i][j] - y[i][j]);
			//get average tp divide diff_avg and get how many factors the error rate
			feat_avg[j] += std::abs(y[i][j]);
			feat_diff_avg[j] += diff;
			sample_error += diff;
			feat_mx[j] = std::max(feat_mx[j], diff);
			feat_mn[j] = std::min(feat_mn[j], diff);
		}
		error_per_sample[i] = sample_error;
		sample_mx = std::max(sample_mx, sample_error);
		sample_mn = std::min(sample_mn, sample_error);
	}
	for (size_t i = 0; i < m; ++i) {
		feat_diff_avg[i] /= n;
		feat_avg[i] /= n;
		feat_avg[i] = feat_diff_avg[i] / feat_avg[i];

	}
	std::ofstream ofs("feature_error_stat.csv", std::ofstream::out);
	ofs << "avg difference\t";
	write_vector(feat_diff_avg, ofs);

	ofs << "avg diff divided avg_y_targ\t";
	write_vector(feat_avg, ofs);

	ofs << "max diff\t";
	write_vector(feat_mx, ofs);

	ofs << "min diff\t";
	write_vector(feat_mn, ofs);
	ofs.close();


	std::ofstream ofs_sample("sample_error_stat.csv", std::ofstream::out);
	ofs_sample << "l1 error\t";
	write_vector(error_per_sample, ofs_sample);

	ofs_sample << "max error amang samples\t" << sample_mx << "\n";
	ofs_sample << "min error amang samples\t" << sample_mn << "\n";
	ofs_sample.close();
}
void get_max_min(const tensor_t & data, vec_t & mn, vec_t & mx) {
	auto n = data.size();
	auto m = data[0].size();
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < m; ++j) {
			mx[j] = std::max(mx[j], data[i][j]);
			mn[j] = std::min(mn[j], data[i][j]);
		}
	}
	for (size_t j = 0; j < m; ++j) {
		mx[j] += std::abs(mx[j]) * 0.07;
		mn[j] -= std::abs(mn[j]) * 0.07;
	}
}
void normalize(tensor_t & data, vec_t & mn, vec_t & mx) {
	auto n = data.size();
	auto m = data[0].size();
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < m; ++j) {
			data[i][j] = (data[i][j] - mn[j]) / (mx[j] - mn[j]);
		}
	}
}
void denormalize(tensor_t & data, vec_t & mn, vec_t & mx) {
	auto n = data.size();
	auto m = data[0].size();
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < m; ++j) {
			data[i][j] = (mx[j] - mn[j]) * data[i][j] + mn[j];
		}
	}
}
tensor_t CsvParser(std::istream& str)
{
	tensor_t result;
	vec_t tmp;
	std::string line;
	for (int i = 0; std::getline(str, line); ++i) {
		result.push_back(tmp);
		std::stringstream lineStream(line);
		std::string cell;

		while (std::getline(lineStream, cell, ' ')){
			if (cell != "") {
				result[i].push_back(stof(cell));
			}
		}
	}
	return result;
}
//#include <conio.h>
void add_n_sample(const tensor_t & scr_data, const int ind, const int cnt, tensor_t & data) {
	for (int i = ind; i < cnt; ++i) {
		data.push_back(scr_data[i]);
	}
}
double square_error(const tensor_t & y, const tensor_t & y_pred) {
	double error = 0.0;
	for (int i = 0; i < y.size(); ++i) {
		for (int j = 0; j < y[i].size(); ++j) {
			error += (y[i][j] - y_pred[i][j])*(y[i][j] - y_pred[i][j]);
		}
	}
	return error / (y.size());
}
int main(int argc, char *argv[]) {
	//_getch();
	unsigned int l1_sz = 40;
	unsigned int l2_sz = 40;
	unsigned int epoch_cnt = 1000;
	if (argc < 4) {
		std::cerr << "train x and y, test y files must be passed\n";
		return -1;
	}
	char *train_x_file = argv[1];
	char *train_y_file = argv[2];
	char *test_x_file = argv[3];

	std::filebuf train_x_file_buf;
	train_x_file_buf.open(train_x_file, std::ios::in);
	std::istream train_x_istream(&train_x_file_buf);

	std::filebuf train_y_file_buf;
	train_y_file_buf.open(train_y_file, std::ios::in);
	std::istream train_y_istream(&train_y_file_buf);

	std::filebuf test_x_file_buf;
	test_x_file_buf.open(test_x_file, std::ios::in);
	std::istream test_x_istream(&test_x_file_buf);

	tensor_t train_data_x = CsvParser(train_x_istream);
	tensor_t label = CsvParser(train_y_istream);
	tensor_t test_data = CsvParser(test_x_istream);

	unsigned int feature_cnt = train_data_x[0].size();

	vec_t in_min(feature_cnt, FLT_MAX);
	vec_t out_min(feature_cnt, FLT_MAX);
	vec_t in_max(feature_cnt, FLT_MIN);
	vec_t out_max(feature_cnt, FLT_MIN);
	get_max_min(train_data_x, in_min, in_max);
	get_max_min(test_data, in_min, in_max);
	get_max_min(label, out_min, out_max);

	normalize(train_data_x, in_min, in_max);
	normalize(label, out_min, out_max);
	normalize(test_data, in_min, in_max);
	write_tensor("train_x_nomr.csv", train_data_x);
	write_tensor("label_nomr.csv", label);
	write_tensor("test_data_nomr.csv", test_data);

	network<sequential> net;
	
	tensor_t train_x_batch;
	tensor_t label_batch;

	adagrad optimizer;
	int cnt = 2;
	freopen("error_static.out", "wt", stdout);
	for (int i = 0; ; ) {
		net = make_mlp<sigmoid>({ feature_cnt, l1_sz, l2_sz/*, l2_sz, l1_sz*/, feature_cnt });
		net.weight_init(weight_init::lecun());
		net.bias_init(weight_init::xavier(2.0));
		add_n_sample(train_data_x, i, cnt, train_x_batch);
		add_n_sample(label, i, cnt, label_batch);
		i = cnt;
		unsigned int batch_sz = std::min(1000, (int)train_x_batch.size());
		net.train<mse>(optimizer, train_x_batch, label_batch, batch_sz, epoch_cnt);
		tensor_t result;
		for (auto &it : train_x_batch) {
			auto one_d_result = net.predict(it);
			result.push_back(one_d_result);
		}
		double error = square_error(label_batch, result);
		std::cout << error << " cnt is " << cnt << "\n";
		std::cerr << error << " cnt is " << cnt << "\n";
		if (cnt >= train_data_x.size() - 1)
		{
			break;
		}
		cnt = std::min(cnt << 1, (int)train_data_x.size() - 1);

	}
	net.save("model");
	tensor_t result;
	for (auto &it : test_data) {
		auto one_d_result = net.predict(it);
		result.push_back(one_d_result);
	}
	denormalize(result, out_min, out_max);
	std::cout << result.size() << std::endl;
	write_tensor("out.csv", result);
	if (argc > 4){
		fprintf(stderr, "Test target file was also passed. Calculate error statistics\n");
		char *test_y_file = argv[4];
		std::filebuf test_y_file_buf;
		test_y_file_buf.open(test_y_file, std::ios::in);
		std::istream test_y_istream(&test_y_file_buf);
		tensor_t test_y = CsvParser(test_y_istream);
		get_error_stat(result, test_y);
	}
	return 0;
}