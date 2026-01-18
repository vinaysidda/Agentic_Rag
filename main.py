# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List

# app = FastAPI()

# # class Students(BaseModel):
# #     id: int
# #     name: str
# #     age: int
# #     grade: str



# # student:list[Students] = [] # instread of dict use Students model like this 
# # next_id = 1 # To auto-increment student IDs

# # # # Uncomment the following code to add item management functionality 

# # @app.get("/student/")
# # def list_students():
# #     return Students


# # @app.get("/student/{student_id}")
# # def get_student(student_id: int):
# #     for stu in student:
# #         if stu["id"] == student_id:
# #             return stu
# #     return {"error": "Student not found"}



# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List

# app = FastAPI()

# class Employee(BaseModel):
#     id: int
#     name: str
#     position: str
#     salary: float


# employees:list[Employee]= [] # instread of dict use Employee model like this

# next_id = 1 # To auto-increment employee IDs

# @app.get("/employee/",response_model=list[Employee])
# def list_employees():
#     return employees

# @app.get("/employee/{employee_id}",response_model=Employee)
# def get_employee(employee_id: int):
#     for emp in employees:
#         if emp["id"] == employee_id:
#             return emp
#     return {"error": "Employee not found behanchod "}


# @app.post("/employee/")
# def create_employee(emp: Employee):
#     global next_id
#     # record = {"id": next_id, "name": emp.name, "position": emp.position, "salary": emp.salary}
#     emp.id = next_id
#     next_id += 1
#     emp.name = emp.name
#     emp.position = emp.position
#     emp.salary = emp.salary

#     # employees.append(emp.dict())
#     employees.append(emp.model_dump())
#     return emp

# @app.put("/employee/{employee_id}")

# def update_employee(employee_id:int,emp:Employee):
#     for index, existing_emp in enumerate(employees):
#         # remeber this  for emp in employees: 
#                     #   print(emp)
#                     # you will just get dict not index 
#     # but for index here you will get
# #     # Loop round	index	existing_emp	Check condition
# # 	0	{"id": 1, "name": "Alice", ...}	existing_emp["id"] == 2 → ❌ False
# # 	1	{"id": 2, "name": "Bob", ...}	existing_emp["id"] == 2 → ✅ True
# # 	2	Won’t reach here (loop stops after match) 


#         if existing_emp["id"] == employee_id:
#             emp.id = employee_id
#             employees[index] = emp.model_dump()
#             return emp
#     return {"error": "Employee not found"}


# @app.delete("/employee/{employee_id}")
# def delete_Employee(employee_id:int):
#     for index , emp in enumerate(employees):
#         if emp["id"] == employee_id:
#             del employees[index]
#             return {"message": "Employee deleted successfully"}
#     return {"error": "Employee not found"}


