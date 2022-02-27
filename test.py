import submission as sb

power_plant = sb.set_probability(sb.make_power_plant_net())

print(sb.get_alarm_prob(power_plant))
print(sb.get_gauge_prob(power_plant))
print(sb.get_temperature_prob(power_plant))
# games_net = sb.get_game_network()
# initial_state = None
# # print(sb.compare_sampling(games_net, initial_state))
# # print(sb.MH_sampler(games_net, initial_state))
# # list_g, list_mh = list(), list()
# # for i in range(100):
# #     data = sb.compare_sampling(games_net, initial_state)
# #     list_g.append(data[2])
# #     list_mh.append(data[3])
# # print(sum(list_g)/sum(list_mh))
# print(sb.calculate_posterior(games_net))
