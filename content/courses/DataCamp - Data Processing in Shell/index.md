---
title: DataCamp - Data Processing in Shell
author: Veera Vignesh
date: '2020-05-02'
slug: datacamp-data-processing-in-shell
categories:
  - Data Science
tags:
  - Tools
subtitle: ''
summary: 'Course Notes'
authors: []
lastmod: '2020-05-02T18:13:57+05:30'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

## Course Overview

- Downloading Data on the Command Line
- Data Cleaning and Munging on the Command Line
- Database Operations on the Command Line
- Data Pipeline on the Command Line

---

### Revisit

[Introduction to Bash Scripting](https://www.notion.so/Introduction-to-Bash-Scripting-4a762e9990724fbbafc462f0566bf3e6)

[Introduction to Shell](https://www.notion.so/Introduction-to-Shell-b70881d7ee0f4f0a9c70c3eeb6c90e10)

---

- Downloading Data on the Command Line
    - curl
        - **C**lient for **URL**s
        - Transfer data to and from the server
        - Accepts many form of request HTTP, HTTPS, FTP ( File Transfer Protocol) , SFTP
        - `curl [option flags...] [URL]`
        - Save file in the same name as the url `-O`
        - Save file with different name `-o newfilename`
        - Can use wildcards to download multiple files at once
        - Using globbling parser to download the files `curl -O https://www.websitename.com/datafilename[001-100].txt`
        - Download every 10th file `curl -O https://www.websitename.com/datafilename[001-100:10].txt`
        - Handling time out  ⇒ redirects when HTTP 300 error occurs`-L`
        - Resuming download `-C`
    - wget
        - Name derived from `world wide web` and `get`
        - `wget [OPTION]... [URL]...`
        - More general purpose then curl can be used to download not only file but also web pages, folders etc
        - Possible to multiple file downloads recurrsively
        - Options unique to wget
            - `-b` - Go to background immediately after startup
            - `-q` - quiet, No output is printed
            - `-c` - resume unfinished downloads
        - Options can be chained under a flag `-bqc`
        - pid (process id) is assigned when we are downloading
        - Downloading in  background generates a log `wget-log` which can be checked to know the status of the process.
    - Advanced wget
        - URL from  local or external file `-i` ⇒ no option flag between `-i` and `file`
        - limit rate for bandwidth optimization `--limit-rate=200k`
        - Mandatory wait secs between requests  `—wait=2.5`
        - Can handle multiple file downloads
        - Can be used to download anything not just HTML
- Data Cleaning and Munging on the Command Line
    - csvkit
        - Is a suite of command-line tools developed in python by WireServices
        - `pip install csvkit`
        - Converting files to csv

            `in2csv —help` or `in2csv -h`

            `in2csv filename.xlsx > filename.csv`

            - Redirect is important or else it will just print out the contents of the excel file
            - Sheet content will be displayed one on which the file was saved.
            - Suppose if there are 2 sheets. `sheet1` and `sheet2` content of sheet2 will be printed if the file was saved on it before closing the file
        - Display contents of specific excel sheet to csv
            - Get the list of sheets inside the excel `in2csv --names filename.xlsx` or just `-n`
            - Print content of `sheet2` → `in2csv —sheet "sheet2" filename.xlsx`
        - View the csv content `csvlook`

            `csvlook filename.csv`

        - Get stats of the csv `csvstat`

            `csvstat filename.csv`

    - Filtering data with csvkit
        - **Column filtering `csvcut`**
            - `csvcut` we can either use column name or column position to filter out the contents of the csv file
            - using `csvcut —n filename.csv` prints out all the column names with their positions
        - **Row filtering `csvgrep`**
            - csvgrep should be followed by one of the following flags
            - `-m` exact match of the item that follows
            - `-r` regex match of the item that follows
            - `-f` followed by a path to the file
            - `csvgrep -c "columnname" -m value filename.csv`
            - All the columns data will be displayed where the value of `columname` is matching the given `value`
    - Stacking data and chaining commands with csvkit
        - Merging two csvfiles `csvstack`
        - Same schema but contents are in different file
        - Keep track of file from which the row came `-g` Shortcode for the filename that we are passing in the same order in which files are passed
        - Rename the column group as something else

            ```bash
            #Basic
            csvstack file1.csv file2.csv > fullfile.csv

            # To find the source of data >> Adds a column called group and denotes
            # From where the observation came from
            csvstack -g "file1","file2" file1.csv file2.csv > fullfile.csv

            #Change the name of Column
            csvstack -g "file1","file2" -n "source" file1.csv file2.csv > fullfile.csv
            ```

        - Chaining of commands

        ```bash
        ; #Links commands together and runs in one line
        && #Links command together, but runs 2nd command only when the first command succeeds
        > # Redirects the output of the first command to the right
        | # Uses output of the first command as input to the second
        ```

- Database Operations on the Command Line
    - Pulling data from database
        - `sql2csv`
        - Allows execution of query on  a large variety of sql databases
        - Pulls data from sql and converts it into csv
        - `sql2csv --db connection_to_db --query SQL_query > csvfilename.csv`
        - Both `--db` and `--query` should be string
        - The full  `—-query` should be all in one line
        - with sql2csv the usage of database management tools can be avoided

        ```bash
        sql2csv --db "sqlite:///SpotifyDatabase.db" \
                --query "SELECT * FROM Spotify_Popularity LIMIT 5" \
                > Spotify_Popularity_5Rows.csv
        ```

        - `sqlite:///` is used while connecting to sqlite
    - Manipulating data using SQL Database
        - `csvsql` - Use SQL to do SQL like syntax on the csv to manipulation
        - creates an in memory sql table to do operation on one or more csv
        - Not suitable for large file processing
        - `csvsql --query "SELECT * FROM TABLENAME LIMIT 1" LOCATION_TO_TABLE`
        - Write the table name in sql query as thought the table is loaded.
        - Can be piped to csvlook or to another csv file
        - Multiple files location can be passed and referenced in the same order that it appeared in the query
        - SQL query can be stored as a variable in bash and can be passed into query
        - Bash allows 80 characters per line
        - To avoid overflow use `\` at the end of the line to notify the command is still on the next line

        ```bash
        # Preview CSV file
        ls

        # Store SQL query as shell variable
        sqlquery="SELECT * FROM Spotify_MusicAttributes ORDER BY duration_ms LIMIT 1"

        # Apply SQL query to Spotify_MusicAttributes.csv
        csvsql --query "$sqlquery" Spotify_MusicAttributes.csv
        ```

        - Joining to csv files using SQL

        ```bash
        # Store SQL query as shell variable
        sql_query="SELECT ma.*, p.popularity FROM Spotify_MusicAttributes ma INNER JOIN Spotify_Popularity p ON ma.track_id = p.track_id"

        # Join 2 local csvs into a new csv using the saved SQL
        csvsql --query "$sql_query" Spotify_MusicAttributes.csv Spotify_Popularity.csv > Spotify_FullData.csv

        # Preview newly created file
        csvstat Spotify_FullData.csv
        ```

    - Pushing data back to the Database
        - `csvsql` with other parameters can be used to upload the csv file to SQL Databases.
        - using `--insert`  `--db`  `--no-inference` `--no-constraints`
        - `--insert`  can be used along with `--db` because we need to have a db to insert into it
        - `csvsql --db "sqlite:///Database_Name.db" --insert Sample.csv`
        - CSV sql takes the data from sample.csv and uploads in the provided database.
        - Under the hood csvkit makes guesses on what these data types  should be and which is the primary key etc. which is needed to insert data into the database
        - In order to change the behavior of  csvsql we can pass in `--no-inference` so csvsql will consider everything as text.
        - Similarly `--no-constraints` allow to insert data into a column without any character limits.

### References:

[csvkit 1.0.5 - csvkit 1.0.5 documentation](https://csvkit.readthedocs.io/en/latest/)