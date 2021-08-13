We use the trained model (model.txt) to make recommendations .
Use "python run.py" to run the falsk api , then try to send post requests to "http://127.0.0.1:5000/recommend"
The request body should be something like that 
{
"age": 45,
"gender": "Male",
"is_active": 1,
"income": 3000000,
"seniority": 50
}
Incomplete json (1 or more missing values) or even empty json {} are allowed and handled (in such cases we just set all values to their default)
