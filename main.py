"""
Script principal pour exécuter le backtest des stratégies obligataires.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

from data.data_loader import DataLoader
from models.laddering_strategy import LadderingStrategy
from models.barbell_strategy import BarbellStrategy
from models.bullet_strategy import BulletStrategy
from backtesting.backtest_engine import BacktestEngine
from analysis.economic_cycle_analyzer import EconomicCycleAnalyzer
from visualization.performance_visualizer import PerformanceVisualizer


def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
        argparse.Namespace: Arguments parsés.
    """
    parser = argparse.ArgumentParser(description='Backtest des stratégies obligataires')
    
    parser.add_argument('--start_date', type=str, default=None,
                      help='Date de début au format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=None,
                      help='Date de fin au format YYYY-MM-DD')
    parser.add_argument('--rebalance_frequency', type=str, default='M',
                      choices=['D', 'W', 'M', 'Q', 'Y'],
                      help='Fréquence de rééquilibrage (D, W, M, Q, Y)')
    parser.add_argument('--initial_capital', type=float, default=1000000,
                      help='Capital initial')
    parser.add_argument('--api_key', type=str, default=None,
                      help='Clé API FRED (optionnelle)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Répertoire pour les résultats')
    parser.add_argument('--benchmark', type=str, default='AGG',
                      help='ETF de référence')
    
    return parser.parse_args()


def main():
    """
    Fonction principale pour exécuter le backtest.
    """
    # Parsing des arguments
    args = parse_args()
    
    # Définir les dates par défaut si non spécifiées
    if not args.start_date:
        args.start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
        
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Afficher les paramètres de backtest
    print(f"\n{'='*80}")
    print(f"BACKTEST DES STRATÉGIES OBLIGATAIRES")
    print(f"{'='*80}")
    print(f"Date de début: {args.start_date}")
    print(f"Date de fin: {args.end_date}")
    print(f"Fréquence de rééquilibrage: {args.rebalance_frequency}")
    print(f"Capital initial: ${args.initial_capital:,.2f}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Répertoire de sortie: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Initialiser le chargeur de données
    data_loader = DataLoader(api_key=args.api_key)
    
    # Charger les données obligataires
    print("Chargement des données obligataires...")
    etf_data = data_loader.load_bond_etfs(start_date=args.start_date, end_date=args.end_date)
    treasury_yields = data_loader.load_treasury_yields(start_date=args.start_date, end_date=args.end_date)
    
    # Initialiser les stratégies
    print("Initialisation des stratégies...")
    laddering = LadderingStrategy(
        name="Laddering Strategy",
        initial_capital=args.initial_capital,
        rebalance_frequency=args.rebalance_frequency
    )
    
    barbell = BarbellStrategy(
        name="Barbell Strategy",
        short_weight=0.5,
        long_weight=0.5,
        initial_capital=args.initial_capital,
        rebalance_frequency=args.rebalance_frequency
    )
    
    bullet = BulletStrategy(
        name="Bullet Strategy (5Y)",
        target_maturity=5,
        initial_capital=args.initial_capital,
        rebalance_frequency=args.rebalance_frequency
    )
    
    bullet_long = BulletStrategy(
        name="Bullet Strategy (10Y)",
        target_maturity=10,
        initial_capital=args.initial_capital,
        rebalance_frequency=args.rebalance_frequency
    )
    
    # Initialiser le moteur de backtest
    backtest_engine = BacktestEngine(
        data_loader=data_loader,
        strategies=[laddering, barbell, bullet, bullet_long],
        benchmark_ticker=args.benchmark
    )
    
    # Exécuter le backtest
    print("\nExécution du backtest...")
    results = backtest_engine.run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        rebalance_frequency=args.rebalance_frequency
    )
    
    # Afficher le résumé des performances
    print("\nRésumé des performances:")
    performance_summary = backtest_engine.get_performance_summary()
    print(performance_summary)
    
    # Sauvegarder les résultats
    print(f"\nSauvegarde des résultats dans {args.output_dir}...")
    performance_summary.to_csv(f"{args.output_dir}/performance_summary.csv")
    
    # Analyse des drawdowns
    drawdown_analysis = backtest_engine.analyze_drawdowns(min_length=20, min_depth=0.05)
    
    # Sauvegarder l'analyse des drawdowns
    for strategy, drawdowns in drawdown_analysis.items():
        if drawdowns:
            drawdowns_df = pd.DataFrame(drawdowns)
            drawdowns_df.to_csv(f"{args.output_dir}/drawdowns_{strategy.replace(' ', '_')}.csv")
    
    # Initialiser le visualiseur de performances
    print("\nCréation des visualisations...")
    visualizer = PerformanceVisualizer()
    
    # Extraire les données des portefeuilles
    portfolios = {name: result['portfolio'] for name, result in results.items()}
    benchmark_data = portfolios.pop(f"Benchmark ({args.benchmark})")
    
    # Créer les visualisations
    fig1 = visualizer.plot_portfolio_value(portfolios, benchmark_data)
    fig1.savefig(f"{args.output_dir}/portfolio_value.png")
    
    fig2 = visualizer.plot_drawdown(portfolios, benchmark_data)
    fig2.savefig(f"{args.output_dir}/drawdowns.png")
    
    fig3 = visualizer.plot_rolling_metrics(portfolios, benchmark_data, metric='returns', window=252)
    fig3.savefig(f"{args.output_dir}/rolling_returns.png")
    
    fig4 = visualizer.plot_rolling_metrics(portfolios, benchmark_data, metric='volatility', window=252)
    fig4.savefig(f"{args.output_dir}/rolling_volatility.png")
    
    fig5 = visualizer.plot_rolling_metrics(portfolios, benchmark_data, metric='sharpe', window=252)
    fig5.savefig(f"{args.output_dir}/rolling_sharpe.png")
    
    fig6 = visualizer.plot_returns_comparison(portfolios, benchmark_data, period='M')
    fig6.savefig(f"{args.output_dir}/monthly_returns.png")
    
    # Si une clé API FRED est fournie, analyser les cycles économiques
    if args.api_key:
        print("\nAnalyse des cycles économiques...")
        cycle_analyzer = EconomicCycleAnalyzer(data_loader)
        
        # Charger les données économiques
        economic_data = cycle_analyzer.load_economic_data(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Identifier les cycles économiques
        cycles = cycle_analyzer.identify_cycles(method='composite', window=12, smoothing=3)
        
        # Analyser la performance par cycle
        cycle_analysis = backtest_engine.analyze_performance_by_economic_cycle(cycles)
        
        # Sauvegarder l'analyse des cycles
        cycle_analysis.to_csv(f"{args.output_dir}/performance_by_cycle.csv")
        
        # Visualiser les cycles
        fig7 = cycle_analyzer.plot_cycles()
        fig7.savefig(f"{args.output_dir}/economic_cycles.png")
        
        # Analyser les rendements des obligations par cycle économique
        bond_performance_by_cycle = cycle_analyzer.analyze_bond_performance_by_cycle(etf_data)
        
        # Visualiser la performance des obligations par cycle
        fig8 = cycle_analyzer.plot_bond_performance_by_cycle(etf_data)
        fig8.savefig(f"{args.output_dir}/bond_performance_by_cycle.png")
        
        # Identifier les régimes de la courbe des taux
        yield_regimes = cycle_analyzer.identify_yield_curve_regime(treasury_yields)
        
        # Analyse combinée des cycles et de la courbe des taux
        combined_analysis = cycle_analyzer.generate_combined_analysis(etf_data, treasury_yields)
        
        # Sauvegarder les recommandations
        recommendations = pd.DataFrame.from_dict(combined_analysis['recommendations'], orient='index')
        recommendations.to_csv(f"{args.output_dir}/strategy_recommendations.csv")
        
        # Afficher les recommandations actuelles
        current_cycle = combined_analysis['current_cycle']
        current_regime = combined_analysis['current_regime']
        
        print(f"\nCycle économique actuel: {current_cycle}")
        print(f"Régime de courbe des taux actuel: {current_regime}")
        
        current_combo = f"{current_cycle}_{current_regime}"
        if current_combo in combined_analysis['recommendations']:
            rec = combined_analysis['recommendations'][current_combo]
            print("\nRecommandation de stratégie actuelle:")
            print(f"  Type: {rec['strategy_type']}")
            print(f"  Justification: {rec['rationale']}")
            print(f"  Meilleur ETF: {rec['best_etf']}")
            print(f"  Rendement attendu: {rec['expected_return']:.2f}%")
            print(f"  Volatilité attendue: {rec['expected_volatility']:.2f}%")
            print(f"  Ratio de Sharpe: {rec['sharpe_ratio']:.2f}")
    
    print(f"\nAnalyse terminée. Les résultats ont été sauvegardés dans {args.output_dir}/")


if __name__ == "__main__":
    main()
