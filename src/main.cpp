#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

# define M_PI           3.14159265358979323846

cv::Mat BoxFilter(const cv::Mat& input) {
	cv::Mat output(input.size(), input.type());
	const auto width = input.cols;
	const auto height = input.rows;

	const int window_size = 5;
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			int sum = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					sum += input.at<uchar>(r + i, c + j);
				}
			}
			output.at<uchar>(r, c) = sum / (window_size * window_size);
		}
	}
	return output;
}

cv::Mat CreateGaussianKernel_2(int window_size) {
	cv::Mat kernel(cv::Size(window_size, window_size), CV_32FC1);

	int half_window_size = window_size / 2;

	// see: lecture_03_slides.pdf, Slide 13
	const double k = 2.5;
	const double r_max = std::sqrt(2.0 * half_window_size * half_window_size);
	const double sigma = r_max / k;

	// sum is for normalization 
	float sum = 0.0;

	for (int x = -window_size / 2; x <= window_size / 2; x++) {
		for (int y = -window_size / 2; y <= window_size / 2; y++) {
			float val = exp(-(x * x + y * y) / (2 * sigma * sigma));
			kernel.at<float>(x + window_size / 2, y + window_size / 2) = val;
			sum += val;
		}
	}

	// normalising the Kernel 
	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5; ++j)
			kernel.at<float>(i, j) /= sum;

	// note that this is a naive implementation
	// there are alternative (better) ways
	// e.g. 
	// - perform analytic normalisation (what's the integral of the gaussian? :))
	// - you could store and compute values as uchar directly in stead of float
	// - computing it as a separable kernel [ exp(x + y) = exp(x) * exp(y) ] ...
	// - ...

	return kernel;
}

cv::Mat CreateGaussianKernel(int window_size, float sigma = 1) // 0.1 ... 3
{
	cv::Mat kernel(window_size, window_size, CV_32FC1);
	double sum = 0.0;
	double i, j;
	for (i = 0; i < window_size; i++) {
		for (j = 0; j < window_size; j++) {
			kernel.at<float>(i, j) = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
			sum += kernel.at<float>(i, j);
		}
	}
	for (i = 0; i < window_size; i++) {
		for (j = 0; j < window_size; j++) {
			kernel.at<float>(i, j) /= sum;
		}
	}
	return kernel;
}

cv::Mat GaussianFilter(const cv::Mat& input) {
	cv::Mat output(input.size(), input.type());
	const auto width = input.cols;
	const auto height = input.rows;
	const int window_size = 5;
	cv::Mat gaussianKernel = CreateGaussianKernel(window_size);
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {

			int sum = 0;
			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					sum += input.at<uchar>(r + i, c + j) * gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2);
				}
			}
			output.at<uchar>(r, c) = sum;
		}
	}
	return output;
}

cv::Mat OurFilter_Bilateral(const cv::Mat& input, const int window_size = 5, float spatial_sigma = 1, float spectral_sigma = 5) {
	const auto width = input.cols;
	const auto height = input.rows;
	cv::Mat output(input.size(), input.type());

	cv::Mat gaussianKernel = CreateGaussianKernel(window_size, spatial_sigma); // sigma for the spatial filter (Gaussian, \(w_G\) kernel)

	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	auto d = [](float a, float b) {
		return std::abs(a - b);
	};

	auto p = [](float val, float sigma) { // use of weighting function p : dissimilar pixels get lower weights, preserves strong edges, smooths other regions
		const float sigmaSq = sigma * sigma;
		const float normalization = std::sqrt(2 * M_PI) * sigma;
		return (1 / normalization) * std::exp(-val / (2 * sigmaSq));
	};

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {

			float sum_w = 0;
			float sum = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {

					float range_difference
						= d(input.at<uchar>(r, c), input.at<uchar>(r + i, c + j));

					float w
						= p(range_difference, spectral_sigma) // spectral filter
						* gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2); // spatial filter

					sum
						+= input.at<uchar>(r + i, c + j) * w;
					sum_w
						+= w;
				}
			}

			output.at<uchar>(r, c) = sum / sum_w;

		}
	}
	return output;
}

void Joint_Bilateral(const cv::Mat& input_rgb, const cv::Mat& input_depth, cv::Mat& output, const int window_size = 5, float sigma = 5) {
	// converting the bilateral filter to Guided Joint bilateral filter for guided image upsampling 
	// upsampling a low resolution depth image, guided by an RGB image
	// weights formed using colors (image input_rgb), filtering happens by modifying depth (image input_depth)
	const auto width = input_rgb.cols;
	const auto height = input_rgb.rows;

	cv::Mat gaussianKernel = CreateGaussianKernel(window_size, 0.5); // sigma for the spatial filter (Gaussian, \(w_G\) kernel)
	auto d = [](float a, float b) {
		return std::abs(a - b);
	};

	auto p = [](float val, float sigma) {	// use of weighting function p : dissimilar pixels get lower weights, preserves strong edges, smooths other regions
		const float sigmaSq = sigma * sigma;
		const float normalization = std::sqrt(2 * M_PI) * sigma;
		return (1 / normalization) * std::exp(-val / (2 * sigmaSq));
	};

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {

			float sum_w = 0;
			float sum = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {

					float range_difference
						= d(input_rgb.at<uchar>(r, c), input_rgb.at<uchar>(r + i, c + j)); // using the rgb image with the spectral filter

					float w
						= p(range_difference, sigma) // sigma for the spectral filter (\(f\) in the slides
						* gaussianKernel.at<float>(i + window_size / 2, j + window_size / 2);

					sum
						+= input_depth.at<uchar>(r + i, c + j) * w; // using the depth image with the spatial filter
					sum_w
						+= w;
				}
			}

			output.at<uchar>(r, c) = sum / sum_w;

		}
	}
}

cv::Mat Upsampling(const cv::Mat& input_rgb, const cv::Mat& input_depth) {
	// applying the joint bilateral filter to upsample a depth image, guided by an RGB image -- iterative upsampling
	int uf = log2(input_rgb.rows / input_depth.rows); // upsample factor
	cv::Mat D = input_depth.clone(); // lowres depth image
	cv::Mat I = input_rgb.clone(); // highres rgb image
	for (int i = 0; i < uf; ++i)
	{
		cv::resize(D, D, D.size() * 2); // doubling the size of the depth image
		cv::resize(I, I, D.size());		// resizing the rgb image to depth image size
		Joint_Bilateral(I, D, D, 5, 0.1); // applying the joint bilateral filter with changed size depth and rbg images
	}
	cv::resize(D, D, input_rgb.size()); // in the end resizing the depth image to rgb image size
	Joint_Bilateral(input_rgb, D, D, 5, 0.1); // applying the joint bilateral filter with full res. size images
	return D;
}

double SSD(const cv::Mat& img1, const cv::Mat& img2)
{
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	return ssd;
}

double RMSE(const cv::Mat& img1, const cv::Mat& img2)
{
	int size = img1.rows * img1.cols;
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	return sqrt(mse);
}

double MSE(const cv::Mat& img1, const cv::Mat& img2)
{
	int size = img1.rows * img1.cols;
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	return mse;
}

double PSNR(const cv::Mat& img1, const cv::Mat& img2)
{

	double max = 255;
	int size = img1.rows * img1.cols;
	double ssd = 0;
	double diff = 0;
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			diff = img1.at<uchar>(r, c) - img2.at<uchar>(r, c);
			ssd += diff * diff;
		}
	}
	double mse = (double)(ssd / size);
	double psnr = 10 * log10((max * max) / mse);
	return psnr;
}

long double mean(const cv::Mat& img)
{
	long double sum = 0;
	int size = img.rows * img.cols;
	for (int r = 0; r < img.rows; ++r) {
		for (int c = 0; c < img.cols; ++c) {
			sum += img.at<uchar>(r, c);
		}
	}
	return sum / size;

}

long double variance(const cv::Mat& img)
{
	cv::Mat var_matrix = img.clone();
	long double sum = 0;
	int size = var_matrix.rows * var_matrix.cols;
	long double mean_ = mean(var_matrix);

	for (int r = 0; r < var_matrix.rows; ++r) {
		for (int c = 0; c < var_matrix.cols; ++c) {
			var_matrix.at<uchar>(r, c) -= mean_;
			var_matrix.at<uchar>(r, c) *= var_matrix.at<uchar>(r, c);
		}
	}

	for (int r = 0; r < var_matrix.rows; ++r) {
		for (int c = 0; c < var_matrix.cols; ++c) {
			sum += var_matrix.at<uchar>(r, c);
		}
	}
	return sum / size;
}

double covariance(const cv::Mat& img1, const cv::Mat& img2)
{
	int size = img1.rows * img1.cols;
	long double sum = 0;
	long double mean1 = mean(img1);
	long double mean2 = mean(img2);
	for (int r = 0; r < img1.rows; ++r) {
		for (int c = 0; c < img1.cols; ++c) {
			sum = sum + ((img1.at<uchar>(r, c) - mean1) * (img2.at<uchar>(r, c) - mean2));
		}
	}
	return sum / size;
}

long double SSIM(const cv::Mat& img1, const cv::Mat& img2)
{
	long double ssim = 0;
	long double k1 = 0.01, k2 = 0.03, L = 255;
	long double C1 = (k1 * L) * (k1 * L);
	long double C2 = (k2 * L) * (k2 * L);

	long double mu_x = mean(img1);
	long double mu_y = mean(img2);
	long double variance_x = variance(img1);
	long double variance_y = variance(img2);
	long double covariance_xy = covariance(img1, img2);

	ssim = ((2 * mu_x * mu_y + C1) * (2 * covariance_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (variance_x * variance_x + variance_y * variance_y + C2));
	return ssim;
}


int main(int argc, char** argv) {

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " DATA_FOLDER PATH" << std::endl;
		return 1;
	}

	std::string dataFolderPath = argv[1];
	cv::Mat input_rgb = cv::imread(dataFolderPath + "view5.png", 0);
	cv::Mat input_depth = cv::imread(dataFolderPath + "lowres_depth.png", 0);
	cv::Mat im = cv::imread(dataFolderPath + "lena.png", 0);

	if (im.data == nullptr) {
		std::cerr << "Failed to load image" << std::endl;
	}
	cv::Mat input = im;
	cv::Mat noise(im.size(), im.type());
	uchar mean = 0;
	uchar stddev = 25;
	cv::randn(noise, mean, stddev);
	im += noise;

	cv::Mat upSampled = Upsampling(input_rgb, input_depth); // converting the bilateral filter to Guided Joint bilateral filter for guided image upsampling
	imwrite(dataFolderPath + "upsampled_depth.PNG", upSampled);
	std::vector<float> spatial_sigma = { 0.1, 1, 1.5, 3 }; // choosing four different levels of sigmas for the spatial and four for the spectral filter.
	std::vector<float> spectral_sigma = { 1, 3, 5, 10 }; // testing all combinations, 16 in total, by running the bilateral filter on the image of my choosing
	for (auto i : spatial_sigma) {	// evaluating the filter on an image of my choice
		for (auto j : spectral_sigma) {
			cv::Mat output_bila_our = OurFilter_Bilateral(im, 5, i, j); // implementation of the bilateral filter
			double ssd = SSD(im, output_bila_our);
			double rmse = RMSE(im, output_bila_our);
			double psnr = PSNR(im, output_bila_our);
			long double ssim = SSIM(im, output_bila_our);
			imwrite(dataFolderPath + "/bilateralComparisons/" + "spatial_sigma_" + std::to_string(i) + "_spectral_sigma_"  // saving the resulting images onto disk
				+ std::to_string(j) + "_ssd_" + std::to_string(ssd) + "_rmse_" + std::to_string(rmse) +
				"_psnr_" + std::to_string(psnr) + "_ssim_" + std::to_string(ssim) + ".PNG", output_bila_our);
		}
	}

	return 0;
}