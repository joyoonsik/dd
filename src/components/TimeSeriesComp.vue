<template>
  <div>
    <div class="SectionTitle">
      <h3>Time Series Graphs</h3>
    </div>
    <v-card class="TimeSeriesComp" elevation="4">
        <LineChart v-if="LineChartDataLoaded" :LineChartData="tmsg"/>
        <LineChart2 v-if="LineChartDataLoaded"/>
    </v-card>
  </div>
</template>

<script>
import EventBus from '../EventBus';
import Vue from 'vue';
import LineChart from './LineChart.vue';
import LineChart2 from './LineChart2.vue';
Vue.config.productionTip = false;


Vue.component('child-component', {
  props: ['val'], // 받는 속성 이름을 지정
  mounted() {
    console.log(this.val)
  },
})

export default {
    name: 'TimeSeriesComp',
    // props:['msg'],
    data: () => ({
        LineChartDataLoaded: false,
        tmsg:false
    }),
    created() { EventBus.$on('message', this.onReceive); }, 
    methods: { 
      onReceive(data) { 
      this.tmsg = data;
      this.LineChartDataLoaded=true; 
      } 
    },
    components:{
        LineChart, LineChart2
    }
}

</script>

<style scoped>

.TimeSeriesComp{
    width: 1870px;
    height: 1550px;
    margin-right: auto;
    margin-left: auto;
    margin-top: 25px;
}

.SectionTitle{
    width: 1870px;
    height: 20px;
    margin-left: auto;
    margin-top: 50px;
    color: rgb(128, 126, 126);
}

.TestComp2{
    margin-left: 10%;
}
</style>