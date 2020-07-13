---
title: Getting Started with Tableau
author: Veera Vignesh
date: '2020-05-11'
slug: getting-started-with-tableau
categories:
  - Data Science
tags:
  - Tableau
subtitle: ''
summary: 'Course Notes from Tableau E-Learning'
authors: []
lastmod: '2020-05-11T22:06:10+05:30'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---
## Understanding the Basic Concepts

### Common types of data sets

- dataset or data source or database
- Types of datasources
  - Spreadsheets
  - Cloud
  - Relational Databases
  - Spatial data , Statistical files
- When data is connected in tableau it is automatically assigned a `Role` and `Type`
- **Role** - A column can either be Dimension or Measure
- Dimension holds the Categorical Data or Qualitative
- Measure holds the Numeric Data or Quantitative
- **Type**  of the field either string, integer, date, geographic etc
- If automatically assigned type is not correct it can be changed and the changes will be saved in Tableau data source (`.tds`) as a metadata
- Field or columns or attribute
- Rows or observation or record
- Data inside the field should obey the definition of the it.
- Continuous Measure - Green color
- Discrete Dimension - Blue color
- Data granularity refers to the level of detail for a piece of data, wherever you are looking.
- In charts these row level data are aggregated to a higher level for visualization
- In order to view the underlying data that made up the viz we can right click and `view data`
- By default measure placed in the view are aggregated with `SUM`
- Other aggregations like average, median & count can also be done
- In order to break the aggregation we use the category/dimension in which we want to show the result
- Detail shelf helps in impacting the granularity without adding it to size or shape
- While writing formulas for calculation we need to aggregate at the level of granularity and not on the lowest granularity that is available
- `Pill` - When a dimension or measure is brought into view Tableau creates a pill
- The value of measures depends on the context of the dimension set. In tableau all measures are aggregators
- Most of the times the dimension are discrete and measures are continuous
- A measure can be converted to discrete → Discount % from continuous to discrete
- A measure can be converted to dimension by dragging and dropping into the dimension region
- When a continuous pill is brought into view it creates an axis taking up the entire view
- When a discrete pill is brought into view it forms a label taking up very little space
- The color also depends on continuous or discrete
  - continuous creates a gradient
  - discrete creates a distinct color for each category
- On a map

> If the geographic type allows it, a measure on color defaults to a filled map. A dimension on color defaults to a symbol map. Whether these
> colors are gradients or palettes depends on if the pill is continuous or discrete

- Right click and drag any pill we can select specifically which type of the pill we need

### Reading Common Chart Types

> Visual analytics leverages our pre‑attentive attributes, which are the visual cues humans automatically process with sensory memory. We can notice and interpret these kinds of attributes quickly and without special effort.

- Visualizations leverages the pre-attentive attributes that the humans can process very fast in order to digest the information thrown at us
- Pre-attentive attributes
  - Length
  - Width
  - Orientation
  - Size
  - Color Hue
  - Enclosure
  - Shape
  - Position
  - Grouping
  - Color Intensity
- Key Components while reading a chart
  - Know what elements make up a chart
  - Ask questions about what you see
  - Watch for misleading charts
- Know what elements make up a chart
  - X-Axis - Qualitative
  - Y-Axis - Quantitative
  - Mark - Bar, Point, Line
  - Filter
  - Legend
  - Tool Tip
  - Labels
- Ask questions about what you see
  - What does this Chart represent
  - Does this chart show any particular patterns or trends?
  - Is this all of the data?
  - Is it clear what has been measured, and what the numbers represent?
- Watch for misleading charts
  - Bar charts with an axis not including zero (0)
  - Color Confusion
  - Wrong chart type for the data
- Charts are most effective when related with correct type of data
- Types of Charts
  - Line Chart - Trends in data over time
  - Bar Chart - Compare Data across categories
  - Heat Map - Show the relationship between two factors
  - Highlight Table Chart - Show detailed information on heat map
  - Tree Map - Show hierarchical data as a proportion of a whole
  - Gantt Chart - Show duration over time - Project timeline
  - Bullet Chart - Evaluate performance metric against a goal
  - Scatter Plot - Between 2 quantitative metrics
  - Histogram - To know distribution of a measure
  - Symbol Map Chart - Used for totals
  - Area Map Chart - Used for rates
  - Box and Whisker Chart - Distribution of set of data
- Bar Chart
  - Order of category can be easily identified
  - Axis represents its value
  - Sometimes the Value axis is hidden and the value is added as a text label to remove clutter
- Line Chart
  - Line chart can broken down into multiple lines based on the segment we are concerned about.
- Scatter Plots
  - Used to measure correlation between two measures
  - Correlation  doesn't mean causation

### Activity

- Process Flow of Tableau Analysis
  - Connect
  - Analyze
  - Share
- **Connect**
  - Allows to connect to external data sources like common flat files to server files
  - Saved Data Sources Represents the data stored in Tableau Repository for analysis
  - Once the file we need to analyze is connected properties of the file like sheets are displayed
  - Choose the sheet that is to be analyzed
  - SQL joins can be done based on two fields in two different sheets
  - Once the sheet is dragged and dropped or double clicked its added to the sheet region
  - Tableau Automatically assigns the type of the data Number, String, Date etc
  - Displays all the field names
  - Once the data is connected any changes made to the data doesn't affect the original data source instead it is stored as a metadata in tableau data source (`.tds`)
- Analyze
  - Crosstab
  - Basic Charts
  - Show Me to select chart type
  - 
- Share
  - `twb` - Workbook without data source
  - `twbx` - Packaged workbook with data