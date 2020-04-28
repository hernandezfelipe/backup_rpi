import os

# mo_tf.py path in Linux
mo_tf_path = '/opt/intel/openvino_2020/deployment_tools/model_optimizer/mo_tf.py'

pb_file = './model/plate_model.pb'
output_dir = './model'
input_shape = [1,64,128,1]
input_shape_str = str(input_shape).replace(' ','')

#os.system("{} --input_model {} --output_dir {} --input_shape {} --scale {} {} {} {} {}".format(mo_tf_path, pb_file,output_dir, input_shape_str, '255','--output dense_4/Sigmoid', '--input conv2d_1_input', '--data_type FP32', '--generate_deprecated_IR_V7'))


os.system("{} --input_model {} --output_dir {} --input_shape {} --scale {} {} {} {} {} {} {}".format(mo_tf_path, pb_file,output_dir, input_shape_str, '255','--output dense_4/Sigmoid', '--input conv2d_1_input', '--data_type FP32', '--disable_fusing', '--disable_gfusing', '--generate_deprecated_IR_V7'))
