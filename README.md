<h1> RAG Based build using Gcp </h1></br>
1.Write the code in order to retrive the files and answer the query in app.py</br>
2.Create the Dockerfile</br>
3.Create the Requiremts file.</br>
4.push all these files to a git repository</br>
5.Login to gcp -> go to cloud storage and create a bucket to store the input documents -> now go to cloud build -> go to triggers -> create a trigger by linking the cloud build to the repository which conists of all the files.</br>
6. when ever there is change made in repo a new build will generate and cloud run will create a api and use that api and paste in postman to see the ask the questions related to documents.
