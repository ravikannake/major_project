


def insert_in_db(name,mobile_num,fine,place,time,vehicle_num):

	my_conn=create_engine("sqlite:////content/drive/MyDrive/my_project/tensorflow-yolov4-tflite/dummy_rto.db")

	query = """
	result = my_conn.execute(ins)