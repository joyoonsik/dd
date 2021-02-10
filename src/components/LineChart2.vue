<template>
<div>
    <div id="my_dataviz2"></div>
</div>
</template>

<script>
import * as d3 from "d3";
import jsondata1 from "../assets/D3LineChart1.json";
import jsondata2 from "../assets/D3LineChart2.json";


export default {
    name: 'LineChart',  
    props:['LineChartData'],
    data(){
        return{
            data:false
        }
    },
mounted(){
this.data=jsondata1;
this.data2=jsondata2;
    var margin = {
            top: 10,
            right: 900,
            bottom: 30,
            left: 170
        },
        width = 2060 - margin.left - margin.right,
        height = 150 - margin.top - margin.bottom;

    var svg = d3.select("#my_dataviz2")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

        var sumstat = d3.nest()
            .key(function(d) {
                return d.key;
            })
            .entries(this.data);

        var x = d3.scaleLinear()
            .domain([d3.min(this.data, function(d) {
                return +d.ymdhms;
            }), d3.max(this.data, function(d) {
                return +d.ymdhms;
            })])
            .range([0, width]);

        var xscale = d3.scaleTime()
            .domain([new Date(2020, 1, 8), new Date(2020, 1, 14)])
            .range([0, width]);

        var x_axis = d3.axisBottom(xscale)
            .ticks(7)
            .tickFormat(d3.timeFormat("%a"));

        svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(x_axis);

        var y = d3.scaleLinear()
            .domain([d3.min(this.data, function(d) {
                return +d.pow;
            }), d3.max(this.data, function(d) {
                return +d.pow;
            })])
            .range([height, 0]);
        svg.append("g")
            .call(d3.axisLeft(y));

        svg.selectAll(".line")
            .data(sumstat)
            .enter()
            .append("path")
            // .style("opacity", .2)
            .style('stroke', '#3598fe')
            .style('fill', 'none')
            .style('stroke-width', '2')
            .attr("fill", "none")
            .attr("stroke-width", 1.5)
            .attr("d", function(d) {
                return d3.line()
                    .x(function(d) {
                        return x(d.ymdhms);
                    })
                    .y(function(d) {
                        return y(+d.pow);
                    }) (d.values)
            })
        //-------2번째-------
var margin2 = {
    top: 10,
    right: 900,
    bottom: 30,
    left: 170
},
width2 = 2060 - margin2.left - margin2.right,
height2 = 450 - margin2.top - margin2.bottom;

var svg2 = d3.select("#my_dataviz2")
.append("svg")
.attr("width", width2 + margin2.left + margin2.right)
.attr("height", height2 + margin2.top + margin2.bottom)
.append("g")
.attr("transform",
    "translate(" + margin2.left + "," + margin2.top + ")");

var sumstat2 = d3.nest()
    .key(function(d) {
        return d.key;
    })
    .entries(this.data2);

var x2 = d3.scaleLinear()
    .domain([d3.min(this.data2, function(d) {
        return +d.ymdhms;
    }), d3.max(this.data2, function(d) {
        return +d.ymdhms;
    })])
    .range([0, width2]);

var xscale2 = d3.scaleTime()
    .domain([new Date(2020, 1, 1), new Date(2020, 1, 2)])
    .range([0, width2]);

var x_axis2 = d3.axisBottom(xscale2)
    .ticks(23)
    .tickFormat(d3.timeFormat("%H"));

svg2.append("g")
    .attr("transform", "translate(0," + height2 + ")")
    .call(x_axis2);

var y2 = d3.scaleLinear()
    .domain([d3.min(this.data2, function(d) {
        return +d.pow;
    }), d3.max(this.data2, function(d) {
        return +d.pow;
    })])
    .range([height2, 0]);
svg2.append("g")
    .call(d3.axisLeft(y2));


svg2.selectAll(".line")
    .data(sumstat2)
    .enter()
    .append("path")
    .attr("fill", "none")
    .attr("stroke-width2", 1.5)
    .attr("d", function(d) {
        return d3.line()
            .x(function(d) {
                return x2(d.ymdhms);
            })
            .y(function(d) {
                return y2(+d.pow);
            }) (d.values)
    })
    .style('stroke', '#3598fe')
    .style('fill', 'none')
    .style('stroke-width', '3')

svg2.selectAll("line-circle")
    .data(this.data2)
    .enter().append("circle")
    .attr("class", "data-circle")
    .attr("r", 5)
    .style('fill', '#3598fe')
    .style('stroke', '#fff')
    .attr("cx", function(d) {
        return x2(d.ymdhms);
    })
    .attr("cy", function(d) {
        return y2(+d.pow);
    });

},
};
</script>