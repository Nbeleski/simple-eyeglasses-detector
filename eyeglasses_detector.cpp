#pragma once

#include "svm_model.hpp"

using namespace dlib;

class eyeglasses_detector
{
private:
	typedef matrix<double, 4500, 1> input_type;
	typedef radial_basis_kernel<input_type> kernel_type;
	typedef decision_function<kernel_type> dec_funct_type;
	typedef normalized_function<dec_funct_type> funct_type;
	funct_type learned_function;

	const std::string get_svm_weights();
	matrix<rgb_pixel> extract_eyes_roi(matrix<rgb_pixel> &in, full_object_detection landmarks);

public:
	eyeglasses_detector();
	~eyeglasses_detector() = default;

	bool verify_eyeglasses(matrix<rgb_pixel> &in, full_object_detection landmarks);
};

matrix<rgb_pixel> eyeglasses_detector::extract_eyes_roi(matrix<rgb_pixel> &in, full_object_detection landmarks)

{
	const int dist_olhos = 64;
	const int pad_x = 18;

	const int pad_y_above = 15;
	const int pad_y_below = 30;

	int odx = (landmarks.part(36).x() + landmarks.part(37).x() + landmarks.part(38).x() + landmarks.part(39).x() + landmarks.part(40).x() + landmarks.part(41).x()) / 6;
	int ody = (landmarks.part(36).y() + landmarks.part(37).y() + landmarks.part(38).y() + landmarks.part(39).y() + landmarks.part(40).y() + landmarks.part(41).y()) / 6;
	int oex = (landmarks.part(42).x() + landmarks.part(43).x() + landmarks.part(44).x() + landmarks.part(45).x() + landmarks.part(46).x() + landmarks.part(47).x()) / 6;
	int oey = (landmarks.part(42).y() + landmarks.part(43).y() + landmarks.part(44).y() + landmarks.part(45).y() + landmarks.part(46).y() + landmarks.part(47).y()) / 6;;

	float dist = sqrt(pow(odx - oex, 2) + pow(ody - oey, 2));
	double resize_factor = (double)(dist_olhos / dist);
	double rotate_radian = atan2(oey - ody, oex - odx);

	matrix<rgb_pixel> rotated;
	auto point_converter = rotate_image(in, rotated, rotate_radian);
	auto new_od = point_converter(dpoint(odx, ody));
	int new_x = new_od.x() * resize_factor;
	int new_y = new_od.y() * resize_factor;

	matrix<rgb_pixel> resized;
	resize_image(resize_factor, rotated);

	matrix<rgb_pixel> out;
	extract_image_chip(rotated, dlib::rectangle(new_x - pad_x, new_y - pad_y_above, new_x + (100 - pad_x - 1), new_y + pad_y_below - 1), out);

	//w.set_image(in);
	//w.clear_overlay();
	//w.add_overlay(render_face_detections(landmarks));
	//cin.get();

	return out;
}

eyeglasses_detector::eyeglasses_detector()
{
	std::istringstream svm_weights_iss(get_svm_weights());
	deserialize(learned_function, svm_weights_iss);
}

bool eyeglasses_detector::verify_eyeglasses(matrix<rgb_pixel> &in, full_object_detection landmarks)
{
	auto eyes_crop = extract_eyes_roi(in, landmarks);
	matrix<unsigned char> lbp;
	make_uniform_lbp_image(eyes_crop, lbp);

	input_type input;
	for (int i = 0; i < 4500; ++i)
		input(i) = lbp(i);

	auto res = learned_function(input);
	return res >= 0;
}