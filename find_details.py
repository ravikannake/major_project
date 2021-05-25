# import required modules
from sqlalchemy import create_engine


def getdetails(plate_num):


	my_conn=create_engine("sqlite:////content/drive/MyDrive/my_project/ANPR_Flask/dummy_rto.db")

	query = """select * from vehicle_owner_details where vehicle_reg_num = ?""" 

	r_set=my_conn.execute(query,(plate_num,))

	details = []

	for row in r_set:

		details.append(row[1])
		details.append(row[2])
		details.append(row[3])
	

	print(details)

	return  details
