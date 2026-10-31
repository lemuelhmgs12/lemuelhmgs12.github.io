---
layout: post 
title: Earthquake Tracking Dashboard Using Tableau 
image: "/posts/tableau-map-image.png" 
tags: [Tableau, Data Viz]
---
This Tableau dashboard tracks global earthquake activity across a 30-day period (July,2022).


# Table of Contents

- [00. Project Overview](#project-overview)
- [01. The Data](#the-data)
- [02. The Dashboard](#the-dashboard)
- [03. Insights](#insights)

# 00. Project Overview <a name="project-overview"></a>
The goal of this project is to create an interactive Tableau dashboard that allows users to explore recent earthquake activity across the world.
Rather than presenting the data as a static table, the dashboard allows the user to investigate earthquake frequency, location, magnitude, and regional patterns in a more visual and interactive way.
This is useful because geographic data is often much easier to understand when it can be explored on a map, filtered by date, and supported with simple summary charts.

# 01. The Data <a name="the-data"></a>
The dashboard uses earthquake data covering a 30-day period.
Each record represents an individual earthquake event, with information such as the date and time of the event, the latitude and longitude, the recorded magnitude, and the location.
To make the dashboard more useful, the data can be viewed at both a detailed location level and a broader regional level.
This helps the user understand not only where individual earthquakes occurred, but also how activity compares across wider parts of the world.

# 02. The Dashboard <a name="the-dashboard"></a>
The dashboard includes a global earthquake map, a table showing the top 10 largest earthquakes, a broad regional breakdown, and location-level analysis for frequency, average magnitude, and maximum magnitude.
This allows the user to quickly move from a high-level global view into more specific location-based insights.
Some useful questions this dashboard can help answer include:
Where are earthquakes most frequently being recorded?
Which locations have the highest average magnitude?
Which individual earthquakes had the largest recorded magnitude?
How does earthquake activity differ across broader global regions?

<iframe seamless frameborder="0" src="https://public.tableau.com/views/EarthquakeAnalysis_17934679909140/EarthquakeTracker?:embed=yes&:display_count=yes&:showVizHome=no" width = '1090' height = '900'></iframe>


# 03. Insights <a name="insights"></a>
From this view, we can see that earthquake activity is not evenly distributed across the world.
Some locations show a high frequency of earthquake records, while others stand out more because of their maximum magnitude values.
This is a useful example of how Tableau can be used to turn raw geographic data into an interactive dashboard that helps users explore patterns quickly, rather than relying only on static tables or written summaries.
It also shows how dashboards can combine different views of the same data, including maps, tables, filters, and summary charts, to create a more complete picture for the user.
