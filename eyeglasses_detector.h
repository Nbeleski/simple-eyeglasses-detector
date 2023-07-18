#include "../dlib/image_processing.h"
#include "../dlib/image_io.h"
#include "../dlib/compress_stream.h"
#include "../dlib/base64.h"


class eyeglasses_detector
{
private:
	typedef dlib::matrix<double, 4500, 1> input_type;
	typedef dlib::radial_basis_kernel<input_type> kernel_type;
	typedef dlib::decision_function<kernel_type> dec_funct_type;
	typedef dlib::normalized_function<dec_funct_type> funct_type;
	funct_type learned_function;

	const std::string get_svm_weights();
	dlib::matrix<dlib::rgb_pixel> extract_eyes_roi(dlib::matrix<dlib::rgb_pixel>& in, dlib::full_object_detection landmarks);

public:
	eyeglasses_detector();
	~eyeglasses_detector() = default;

	bool verify_eyeglasses(dlib::matrix<dlib::rgb_pixel>& in, dlib::full_object_detection landmarks);
};